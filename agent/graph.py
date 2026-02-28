"""
agent/graph.py

Full LangGraph agent orchestration:
  Intent Router → Semantic Tool Retriever → LLM Planner → 
  Tool Executor → Validator → Response Synthesizer

LangSmith tracing enabled via environment variables.
"""

import os
import json
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from vectorstore.embed_tools import retrieve_top_tools, search_tool_registry
from tools.paypal_executor import execute_paypal_api
from agent.request_history import (
    ensure_list,
    append_request,
    format_last_requests,
)

load_dotenv()

# Config (env with defaults)
TOOL_TOP_K = int(os.getenv("TOOL_TOP_K", "5"))
TOOL_CONFIDENCE_THRESHOLD = float(os.getenv("TOOL_CONFIDENCE_THRESHOLD", "0.3"))
MAX_MULTI_STEPS = int(os.getenv("MAX_MULTI_STEPS", "5"))
REQUEST_HISTORY_MAX = int(os.getenv("REQUEST_HISTORY_MAX", "20"))

# LangSmith is auto-enabled from env vars:
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=...
# LANGCHAIN_PROJECT=paypal-agent

# ─────────────────────────────────────────
# LLM Setup — Groq with LLaMA 3.1 70B
# ─────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


# ─────────────────────────────────────────
# Agent State
# ─────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    intent: str                  # api_action | knowledge | system_search | multi_step
    retrieved_tools: list        # top-k tools from vector DB
    execution_plan: dict         # LLM's plan: {tool_name, endpoint, method, params}
    api_result: dict             # raw API response
    final_response: str          # formatted answer to user
    retry_count: int
    error: str
    rag_sources: list            # source documents used by RAG
    rag_chunks: list             # chunks retrieved by RAG
    # Optional / extended state (set when used)
    error_type: str              # auth | validation | not_found | rate_limit | server_error | unknown
    request_history: list        # last N requests for "status of last request"
    skip_to_clarification: bool  # true when tool confidence too low
    multi_step_results: list     # accumulated step results for multi_step
    multi_step_step_index: int   # current step (0-based)


# ─────────────────────────────────────────
# Helpers: Structured error classification
# ─────────────────────────────────────────
def _classify_error_type(api_result: dict) -> str:
    """Map API failure to error type for tailored retry/messages."""
    if api_result.get("success"):
        return ""
    code = api_result.get("status_code", 0)
    data = api_result.get("data") or {}
    msg = str(data.get("message", "")).lower() + json.dumps(data).lower()
    if code == 401 or "authentication" in msg or "unauthorized" in msg:
        return "auth"
    if code == 429 or "rate" in msg or "throttl" in msg:
        return "rate_limit"
    if code == 404 or "not found" in msg:
        return "not_found"
    if code in (400, 422) or "validation" in msg or "invalid" in msg:
        return "validation"
    if 500 <= code < 600:
        return "server_error"
    return "unknown"


# ─────────────────────────────────────────
# Node 1: Intent Router
# ─────────────────────────────────────────
def intent_router(state: AgentState) -> AgentState:
    """Classify what type of request the user is making. Uses request_history (memory) when present."""
    query = state["user_query"].strip()
    history = ensure_list(state.get("request_history"))

    memory_block = ""
    if history:
        memory_block = f"""
Session's recent requests (memory):
{format_last_requests(history, last_n=5)}

If the user is asking about the status of their last request, what just happened, or "did it work?", classify as system_search.
"""

    prompt = f"""You are an intent classifier for a PayPal agent.
{memory_block}
Classify this user query into exactly ONE category:
- api_action: user wants to DO something (send payment, create invoice, check order, get disputes)
- knowledge: user asking HOW something works or needs explanation from documents (e.g. "how does refund work?" from docs)
- system_search: user asking WHAT TOOLS/capabilities exist, what the agent can do, or "how do I ... via the API" (e.g. "What tools do I have for subscriptions?", "How do I send an invoice via the API?") OR asking about the status of their last/previous request (use memory above)
- multi_step: user request needs multiple API calls in sequence (e.g. create subscription plan then subscribe, or create invoice then send)

If the user asks "what tools", "what can I do", "how do I ... via the API", or which endpoints/capabilities exist → system_search (they want the tool registry, not RAG).
If the user wants to "create a subscription" or "subscription plan for $X/month" → multi_step (requires: create plan first, then create subscription with that plan_id).

User query: "{query}"

Respond with ONLY one word from: api_action, knowledge, system_search, multi_step"""

    response = llm.invoke([HumanMessage(content=prompt)])
    intent = response.content.strip().lower()

    if intent not in ["api_action", "knowledge", "system_search", "multi_step"]:
        intent = "api_action"

    print(f"🔀 Intent: {intent}")
    return {**state, "intent": intent}


# ─────────────────────────────────────────
# Node 2: Semantic Tool Retriever
# ─────────────────────────────────────────
def tool_retriever(state: AgentState) -> AgentState:
    """Retrieve top-k most relevant tools; if best score below threshold, ask for clarification."""
    query = state["user_query"]
    try:
        top_tools = retrieve_top_tools(query, top_k=TOOL_TOP_K)
    except Exception as e:
        err_msg = str(e).lower()
        if "hnsw" in err_msg or "nothing found on disk" in err_msg or "chroma" in err_msg:
            print("⚠️ ChromaDB tool index missing or corrupted. Run from project root: python ingest.py")
            return {
                **state,
                "retrieved_tools": [],
                "skip_to_clarification": True,
                "error": "Tool registry unavailable. From the project root run: **python ingest.py** to index PayPal tools.",
            }
        raise

    print(f"🔍 Retrieved {len(top_tools)} tools (top_k={TOOL_TOP_K}):")
    for t in top_tools:
        print(f"   → {t['name']} (score: {t['similarity_score']})")

    if not top_tools:
        return {**state, "retrieved_tools": [], "skip_to_clarification": True}
    best_score = top_tools[0].get("similarity_score", 0)
    if best_score < TOOL_CONFIDENCE_THRESHOLD:
        return {
            **state,
            "retrieved_tools": top_tools,
            "skip_to_clarification": True,
        }
    return {**state, "retrieved_tools": top_tools, "skip_to_clarification": False}


# ─────────────────────────────────────────
# Node 3: System Search Tool
# ─────────────────────────────────────────
def system_search_node(state: AgentState) -> AgentState:
    """Handle system/meta queries: tool capabilities OR last request status. Uses memory to decide."""
    query = state["user_query"]
    history = ensure_list(state.get("request_history"))

    # Use memory: if we have history, ask LLM whether user is asking about last request (no hardcoded phrases)
    if history:
        prompt = f'''User query: "{query}"

Session's last requests:
{format_last_requests(history, last_n=5)}

Is the user asking about the status of their last/previous request or what just happened? Reply only YES or NO.'''
        response = llm.invoke([HumanMessage(content=prompt)])
        if response.content.strip().upper().startswith("YES"):
            result = format_last_requests(history)
            h = append_request(history, query, "system_search", True, "System Search", summary=result[:150], max_entries=REQUEST_HISTORY_MAX)
            return {**state, "final_response": result, "request_history": h}

    result = search_tool_registry(query)
    h = append_request(ensure_list(state.get("request_history")), query, "system_search", True, "System Search", summary=result[:150], max_entries=REQUEST_HISTORY_MAX)
    return {**state, "final_response": result, "request_history": h}


# ─────────────────────────────────────────
# Node 2b: Clarification (low confidence in tool match)
# ─────────────────────────────────────────
def clarification_node(state: AgentState) -> AgentState:
    """User intent unclear; ask for clarification instead of guessing."""
    msg = state.get("error") or (
        "I'm not sure which action you mean. Could you rephrase or be more specific? "
        "For example: \"Send an invoice for $50 to john@example.com\", "
        "\"List my invoices\", or \"What tools are available for disputes?\""
    )
    history = append_request(
        ensure_list(state.get("request_history")),
        query=state.get("user_query", ""),
        intent=state.get("intent", ""),
        success=False,
        tool_name="(clarification)",
        error_type="low_confidence",
        summary=msg[:200],
        max_entries=REQUEST_HISTORY_MAX,
    )
    return {**state, "final_response": msg, "request_history": history}


# ─────────────────────────────────────────
# Node 4: RAG Tool (knowledge questions)
# ─────────────────────────────────────────
def rag_tool_node(state: AgentState) -> AgentState:
    """Handle knowledge questions using RAG over uploaded documents."""
    query = state["user_query"]
    try:
        from agent.rag_agent import answer_from_documents
        result = answer_from_documents(query=query)
        ans = result["answer"]
        h = append_request(ensure_list(state.get("request_history")), query, "knowledge", True, "RAG", summary=ans[:150], max_entries=REQUEST_HISTORY_MAX)
        return {
            **state,
            "final_response": ans,
            "rag_sources": result.get("sources", []),
            "rag_chunks": result.get("chunks", []),
            "request_history": h,
        }
    except Exception as e:
        h = append_request(ensure_list(state.get("request_history")), query, "knowledge", False, "RAG", error_type="error", summary=str(e)[:150], max_entries=REQUEST_HISTORY_MAX)
        return {**state, "final_response": f"RAG tool error: {str(e)}", "request_history": h}


# ─────────────────────────────────────────
# Node 5: LLM Planner
# ─────────────────────────────────────────
def llm_planner(state: AgentState) -> AgentState:
    """
    LLM sees ONLY top-k tools and creates an execution plan.
    Uses conversation history and last error (on retry) for context.
    """
    query = state["user_query"]
    tools = state.get("retrieved_tools") or []
    retry_count = state.get("retry_count", 0)
    api_result = state.get("api_result", {})
    error_type = state.get("error_type", "")
    messages = state.get("messages") or []

    tools_description = json.dumps([
        {
            "name": t["name"],
            "method": t["method"],
            "endpoint": t["endpoint"],
            "description": t["description"],
            "parameters": t["parameters"]
        }
        for t in tools
    ], indent=2)

    # Recent conversation (last 2 user + 2 assistant for context)
    conv_lines = []
    for m in messages[-6:]:
        role = getattr(m, "type", None) or (m.get("type") if isinstance(m, dict) else "")
        content = getattr(m, "content", None) or (m.get("content", "") if isinstance(m, dict) else "")
        if role == "human" or (isinstance(m, dict) and m.get("role") == "user"):
            conv_lines.append(f"User: {str(content)[:200]}")
        elif role == "ai" or (isinstance(m, dict) and m.get("role") == "assistant"):
            conv_lines.append(f"Assistant: {str(content)[:150]}...")
    conversation_context = "\n".join(conv_lines) if conv_lines else "(no prior messages)"

    # On retry: include last error so LLM can try different tool/params
    retry_context = ""
    if retry_count > 0 and not api_result.get("success"):
        err_data = api_result.get("data") or {}
        retry_context = f"""
PREVIOUS ATTEMPT FAILED (try #{retry_count}). Do NOT repeat the same call.
Error type: {error_type}
Error response: {json.dumps(err_data)[:400]}
Choose a different tool or fix the parameters and try again.
"""

    # Generate unique invoice number using timestamp
    import time
    invoice_num = str(int(time.time()))[-8:]

    prompt = f"""You are a PayPal API planning agent. Given the user's request and available tools, create an execution plan.

User Request: "{query}"
{retry_context}

Recent conversation (for context):
{conversation_context}

Available Tools (choose the BEST one):
{tools_description}

IMPORTANT RULES:
1. NEVER use endpoints with placeholder IDs like /:invoice_id/ or /:order_id/ unless you already have a real ID
2. If user wants to "send invoice", you must FIRST create it using POST /v2/invoicing/invoices
3. Always start with the CREATE endpoint when the resource doesn't exist yet
4. Extract ALL values from the user's message (amounts, emails, names, dates)
5. Use sandbox base URL: https://api-m.sandbox.paypal.com
6. Return ONLY valid JSON, no markdown, no explanation
7. Subscriptions: To "create a subscription" or "subscription plan for $X/month", use multi_step: FIRST Create plan (POST /v1/billing/plans) with billing_cycles for that price, THEN Create subscription (POST /v1/billing/subscriptions) with the plan_id returned from step 1. Never invent plan_id (e.g. "P-10M"); it must come from the Create plan API response. start_time for Create subscription MUST be a future ISO 8601 datetime (e.g. one hour from now).

For "Send invoice for $50 to john@example.com":
- Correct first step: POST /v2/invoicing/invoices (create it first!)
- Wrong: POST /v2/invoicing/invoices/:invoice_id/send (no ID yet!)

Create a JSON execution plan with this EXACT structure:
{{
  "tool_name": "name of the tool to use",
  "method": "HTTP method",
  "endpoint": "full URL - NO placeholder :param_id values",
  "body": {{
    ...all parameters extracted from user query...
  }},
  "query_params": {{}},
  "reasoning": "why you chose this tool and this step"
}}

EXAMPLE - for "Send invoice for $50 to john@example.com", body must be:
{{
  "detail": {{
    "invoice_number": "INV-{invoice_num}",
    "currency_code": "USD",
    "note": "Thank you for your business"
  }},
  "primary_recipients": [{{
    "billing_info": {{
      "email_address": "john@example.com"
    }}
  }}],
  "items": [{{
    "name": "Service",
    "quantity": "1",
    "unit_amount": {{
      "currency_code": "USD",
      "value": "50.00"
    }}
  }}]
}}

Replace john@example.com and 50.00 with actual values from the user query."""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        # Clean the response and parse JSON
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        plan = json.loads(content)
        print(f"📋 Plan: {plan.get('tool_name')} — {plan.get('reasoning', '')}")
        return {**state, "execution_plan": plan}
    except Exception as e:
        print(f"⚠️  Plan parsing failed: {e}")
        return {**state, "execution_plan": {}, "error": str(e)}


# ─────────────────────────────────────────
# Node 6: Tool Executor
# ─────────────────────────────────────────
def tool_executor(state: AgentState) -> AgentState:
    """Execute the planned API call against PayPal Sandbox."""
    import time
    import json as _json

    plan = state.get("execution_plan", {})

    if not plan or not plan.get("endpoint"):
        return {**state, "api_result": {"success": False, "error": "No valid plan"}}

    # ── Always inject a unique invoice number — never trust LLM to generate one
    body = plan.get("body") or {}
    if "invoicing/invoices" in plan.get("endpoint", "") and plan.get("method", "").upper() == "POST":
        if "detail" in body:
            unique_num = f"INV-{int(time.time())}"
            body["detail"]["invoice_number"] = unique_num
            plan["body"] = body
            print(f"🔢 Injected unique invoice number: {unique_num}")

    # ── Subscription start_time must be a future date; fix if missing or in the past
    if "billing/subscriptions" in plan.get("endpoint", "") and plan.get("method", "").upper() == "POST":
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        future = (now + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        if not body.get("start_time") or body.get("start_time", "") < now.strftime("%Y-%m-%dT%H:%M:%S.000Z"):
            body["start_time"] = future
            plan["body"] = body
            print(f"📅 Injected future start_time for subscription: {future}")

    print(f"⚡ Executing: {plan['method']} {plan['endpoint']}")

    result = execute_paypal_api(
        method=plan.get("method", "GET"),
        endpoint=plan.get("endpoint", ""),
        body=plan.get("body"),
        params=plan.get("query_params")
    )
    result["error_type"] = _classify_error_type(result)
    print(f"📡 API Response: {result['status_code']}" + (f" (error_type={result['error_type']})" if result.get("error_type") else ""))
    return {**state, "api_result": result, "error_type": result.get("error_type", "")}


# ─────────────────────────────────────────
# Node 7: Validator
# ─────────────────────────────────────────
def validator(state: AgentState) -> str:
    """Check if API call succeeded. Route to success, retry, or give_up (e.g. auth errors)."""
    result = state.get("api_result", {})
    retry_count = state.get("retry_count", 0)
    error_type = state.get("error_type", "")

    if result.get("success"):
        return "success"
    # Don't retry auth errors (wrong credentials)
    if error_type == "auth":
        return "give_up"
    if retry_count >= 2:
        return "give_up"
    return "retry"


# ─────────────────────────────────────────
# Node 8: Retry Node
# ─────────────────────────────────────────
def retry_node(state: AgentState) -> AgentState:
    """Increment retry counter and re-retrieve tools."""
    retry_count = state.get("retry_count", 0)
    print(f"🔁 Retrying... attempt {retry_count + 1}")
    return {**state, "retry_count": retry_count + 1}


# ─────────────────────────────────────────
# Node 9: Response Synthesizer
# ─────────────────────────────────────────
def multi_step_decide(state: AgentState) -> AgentState:
    """After a successful step in multi_step intent: append result, ask LLM if another step or done."""
    multi_step_results = list(state.get("multi_step_results") or [])
    step_index = state.get("multi_step_step_index") or 0
    plan = state.get("execution_plan", {})
    api_result = state.get("api_result", {})
    query = state["user_query"]

    multi_step_results.append({
        "step": step_index + 1,
        "tool_name": plan.get("tool_name", ""),
        "result_summary": json.dumps(api_result.get("data", {}))[:300],
    })
    step_index += 1

    if step_index >= MAX_MULTI_STEPS:
        state = {**state, "multi_step_results": multi_step_results, "multi_step_step_index": step_index, "_next_node": "synthesize"}
        return state

    # Ask LLM: one more step or done?
    steps_done = "\n".join([f"  Step {r['step']}: {r['tool_name']} -> {r['result_summary'][:100]}..." for r in multi_step_results])
    last_data = api_result.get("data") or {}
    prompt = f"""User asked: "{query}"

Steps completed so far:
{steps_done}

Last API result (full): {json.dumps(last_data)[:600]}

Is there exactly one more API call needed to fully satisfy the user? If no more steps needed, reply with ONLY: {{"done": true}}
If one more step is needed, reply with a JSON execution plan: {{"done": false, "tool_name": "...", "method": "GET|POST|...", "endpoint": "full URL", "body": {{}} or null, "query_params": {{}}}}
If the next step is Create subscription (POST .../billing/subscriptions), you MUST set body.plan_id to the plan id from the last API result (e.g. last_data["id"] from Create plan response). Set body.start_time to a future ISO 8601 datetime (e.g. one hour from now).
Return ONLY valid JSON, no markdown."""

    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()
    if "```" in content:
        content = content.split("```")[1].replace("json", "").strip()
    try:
        decision = json.loads(content)
    except Exception:
        decision = {"done": True}
    if decision.get("done"):
        return {**state, "multi_step_results": multi_step_results, "multi_step_step_index": step_index, "_next_node": "synthesize"}
    # Next step
    next_plan = {
        "tool_name": decision.get("tool_name", "Next step"),
        "method": decision.get("method", "GET"),
        "endpoint": decision.get("endpoint", ""),
        "body": decision.get("body"),
        "query_params": decision.get("query_params", {}),
        "reasoning": "Multi-step next",
    }
    return {
        **state,
        "multi_step_results": multi_step_results,
        "multi_step_step_index": step_index,
        "execution_plan": next_plan,
        "_next_node": "execute_next",
    }


def response_synthesizer(state: AgentState) -> AgentState:
    """Format the API result into a human-friendly response."""
    query = state["user_query"]
    api_result = state.get("api_result", {})
    plan = state.get("execution_plan", {})
    multi_step_results = state.get("multi_step_results") or []

    # Multi-step summary when we just finished a multi_step flow
    if multi_step_results and state.get("intent") == "multi_step":
        lines = [f"**Multi-step completed ({len(multi_step_results)} step(s)):**\n"]
        for r in multi_step_results:
            lines.append(f"- Step {r['step']}: {r['tool_name']}")
        prompt = f"""User asked: "{query}". We completed {len(multi_step_results)} API step(s). Summarize what was done in a short, friendly response."""
        response = llm.invoke([HumanMessage(content=prompt)])
        final = response.content
        history = append_request(
            ensure_list(state.get("request_history")),
            query=query,
            intent="multi_step",
            success=True,
            tool_name=f"{len(multi_step_results)} steps",
            summary=final[:200],
            max_entries=REQUEST_HISTORY_MAX,
        )
        return {**state, "final_response": final, "request_history": history}

    if api_result.get("success"):
        data = api_result.get("data", {})

        # ── Auto follow-up: if we just CREATED an invoice and user said "send",
        #    automatically send it now using the returned href/id
        invoice_id = data.get("id") or data.get("invoice_id")
        send_keywords = any(w in query.lower() for w in ["send", "invoice for", "bill"])

        if invoice_id and send_keywords and "invoicing/invoices" in plan.get("endpoint", ""):
            print(f"📤 Auto-sending created invoice: {invoice_id}")
            send_result = execute_paypal_api(
                method="POST",
                endpoint=f"/v2/invoicing/invoices/{invoice_id}/send",
                body={"send_to_recipient": True}
            )
            if send_result.get("success"):
                final_response = (
                    f"✅ Invoice created and sent successfully!\n\n"
                    f"**Invoice ID:** `{invoice_id}`\n"
                    f"**Sent to:** {_extract_email(query)}\n"
                    f"**Amount:** {_extract_amount(query)}\n\n"
                    f"The recipient will receive the invoice via email."
                )
                return {**state, "final_response": final_response}

        # Normal success response — use larger limit for list responses so all items are visible
        data_str = json.dumps(data, indent=2)
        is_list_response = isinstance(data.get("items"), list) or "total_items" in data or "total_pages" in data
        max_len = 6000 if is_list_response else 2000
        data_slice = data_str[:max_len]

        list_instructions = ""
        if is_list_response:
            list_instructions = """
For LIST responses (items array): Show each item with ONLY the fields that are actually present in the data.
- If amount/detail/recipient is present for an item, show it.
- If a field is missing for an item, write "—" or omit that line; do NOT say "Not specified in the provided data".
- Prefer a short table or bullet list. Mention total count (e.g. "You have X invoices") and that the user can open the self link for full details if needed."""

        prompt = f"""You are a helpful PayPal assistant. Format this API response for the user.

User asked: "{query}"
API Tool Used: {plan.get('tool_name', 'PayPal API')}
API Response Data:
{data_slice}
{list_instructions}

Write a clear, concise, friendly response. 
Highlight key info: IDs, amounts, status, next steps.
Use markdown formatting with bold for important values."""

    else:
        error_data = api_result.get("data", {})
        prompt = f"""The PayPal API call failed. Explain this to the user helpfully.

User asked: "{query}"
Error: {json.dumps(error_data)[:500]}

Give a clear explanation and suggest what the user can try.
Common issues: missing credentials, invalid parameters, resource not found."""

    response = llm.invoke([HumanMessage(content=prompt)])
    final = response.content
    # Append to request history for "status of last request"
    plan = state.get("execution_plan", {})
    history = append_request(
        ensure_list(state.get("request_history")),
        query=state["user_query"],
        intent=state.get("intent", ""),
        success=True,
        tool_name=plan.get("tool_name", ""),
        summary=final[:200],
        max_entries=REQUEST_HISTORY_MAX,
    )
    return {**state, "final_response": final, "request_history": history}


def _extract_email(text: str) -> str:
    """Extract email from text."""
    import re
    match = re.search(r'[\w.-]+@[\w.-]+\.\w+', text)
    return match.group(0) if match else "recipient"


def _extract_amount(text: str) -> str:
    """Extract amount from text."""
    import re
    match = re.search(r'\$[\d.]+|\d+\s*USD', text)
    return match.group(0) if match else "specified amount"


def failure_response(state: AgentState) -> AgentState:
    """Handle complete failure after retries; message tailored to error_type."""
    error_type = state.get("error_type", "unknown")
    if error_type == "auth":
        final_response = (
            "The request failed due to an **authentication error**. "
            "Please check your PayPal sandbox **Client ID** and **Secret** in `.env` and try again."
        )
    elif error_type == "rate_limit":
        final_response = (
            "PayPal **rate limit** was hit. Please wait a minute and try again."
        )
    elif error_type == "not_found":
        final_response = (
            "The requested resource was **not found**. Check that IDs (invoice, order, etc.) are correct."
        )
    elif error_type == "validation":
        final_response = (
            "The request was **rejected** (invalid parameters or body). "
            "Try rephrasing your request or check the required fields for this action."
        )
    else:
        final_response = (
            "I wasn't able to complete that request after several attempts. "
            "This might be due to invalid credentials or the API being unavailable. "
            "Please check your PayPal sandbox credentials and try again."
        )
    plan = state.get("execution_plan", {})
    history = append_request(
        ensure_list(state.get("request_history")),
        query=state.get("user_query", ""),
        intent=state.get("intent", ""),
        success=False,
        tool_name=plan.get("tool_name", ""),
        error_type=error_type,
        summary=final_response[:200],
        max_entries=REQUEST_HISTORY_MAX,
    )
    return {**state, "final_response": final_response, "request_history": history}


# ─────────────────────────────────────────
# Route after tool_retriever (confidence check)
# ─────────────────────────────────────────
def route_after_retriever(state: AgentState) -> str:
    if state.get("skip_to_clarification"):
        return "clarification"
    return "llm_planner"


# ─────────────────────────────────────────
# Route after success (single vs multi_step)
# ─────────────────────────────────────────
def route_after_success(state: AgentState) -> str:
    if state.get("intent") == "multi_step":
        return "multi_step_decide"
    return "response_synthesizer"


# ─────────────────────────────────────────
# Route after multi_step_decide
# ─────────────────────────────────────────
def route_after_multi_step_decide(state: AgentState) -> str:
    return state.get("_next_node", "synthesize")


# ─────────────────────────────────────────
# Route after Intent Router
# ─────────────────────────────────────────
def route_by_intent(state: AgentState) -> str:
    intent = state.get("intent", "api_action")
    if intent == "system_search":
        return "system_search"
    elif intent == "knowledge":
        return "rag_tool"
    else:
        return "tool_retriever"


# ─────────────────────────────────────────
# Build the LangGraph
# ─────────────────────────────────────────
def build_agent():
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("intent_router", intent_router)
    graph.add_node("tool_retriever", tool_retriever)
    graph.add_node("clarification_node", clarification_node)
    graph.add_node("system_search", system_search_node)
    graph.add_node("rag_tool", rag_tool_node)
    graph.add_node("llm_planner", llm_planner)
    graph.add_node("tool_executor", tool_executor)
    graph.add_node("retry_node", retry_node)
    graph.add_node("multi_step_decide", multi_step_decide)
    graph.add_node("response_synthesizer", response_synthesizer)
    graph.add_node("failure_response", failure_response)

    # Entry point
    graph.set_entry_point("intent_router")

    # Routing from intent
    graph.add_conditional_edges(
        "intent_router",
        route_by_intent,
        {
            "system_search": "system_search",
            "rag_tool": "rag_tool",
            "tool_retriever": "tool_retriever"
        }
    )

    # After tool_retriever: clarification or planner (confidence check)
    graph.add_conditional_edges(
        "tool_retriever",
        route_after_retriever,
        {"clarification": "clarification_node", "llm_planner": "llm_planner"}
    )
    graph.add_edge("clarification_node", END)

    # Linear: plan → execute
    graph.add_edge("llm_planner", "tool_executor")

    # Conditional: validate result. On success, route to after_success_router to branch single vs multi_step.
    # So we need: success -> a router that sends multi_step to multi_step_decide, else to response_synthesizer.
    # We can't do that with validator alone. So: validator success -> "after_success" node that just routes.
    # Add a no-op node "after_success" that returns state, then conditional_edges from it.
    def after_success_router(state: AgentState) -> AgentState:
        return state
    graph.add_node("after_success_router", after_success_router)
    graph.add_conditional_edges(
        "tool_executor",
        validator,
        {
            "success": "after_success_router",
            "retry": "retry_node",
            "give_up": "failure_response"
        }
    )
    graph.add_conditional_edges(
        "after_success_router",
        route_after_success,
        {"multi_step_decide": "multi_step_decide", "response_synthesizer": "response_synthesizer"}
    )

    # Multi-step: either next step (tool_executor) or done (response_synthesizer)
    graph.add_conditional_edges(
        "multi_step_decide",
        route_after_multi_step_decide,
        {"synthesize": "response_synthesizer", "execute_next": "tool_executor"}
    )

    # Retry: re-retrieve tools (and planner will see last error)
    graph.add_edge("retry_node", "tool_retriever")

    # Terminal nodes
    graph.add_edge("response_synthesizer", END)
    graph.add_edge("failure_response", END)
    graph.add_edge("system_search", END)
    graph.add_edge("rag_tool", END)

    return graph.compile()


# Singleton agent
agent = build_agent()


def _build_initial_state(user_query: str, request_history: list = None) -> dict:
    return {
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
        "intent": "",
        "retrieved_tools": [],
        "execution_plan": {},
        "api_result": {},
        "final_response": "",
        "retry_count": 0,
        "error": "",
        "rag_sources": [],
        "rag_chunks": [],
        "request_history": ensure_list(request_history),
        "multi_step_results": [],
        "multi_step_step_index": 0,
        "error_type": "",
        "skip_to_clarification": False,
    }


def run_agent(user_query: str, request_history: list = None) -> str:
    """Run the agent — returns plain string response."""
    result = agent.invoke(_build_initial_state(user_query, request_history))
    return result.get("final_response", "I could not process your request.")


def run_agent_full(user_query: str, request_history: list = None) -> dict:
    """Run the agent — returns full result dict with intent, sources, chunks, request_history."""
    result = agent.invoke(_build_initial_state(user_query, request_history))
    return {
        "final_response": result.get("final_response", "I could not process your request."),
        "intent": result.get("intent", ""),
        "sources": result.get("rag_sources", []),
        "chunks": result.get("rag_chunks", []),
        "request_history": result.get("request_history", []),
    }