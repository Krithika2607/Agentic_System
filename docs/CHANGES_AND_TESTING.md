# Changes Summary & How to Test Them

This document lists all changes made to implement the gap-analysis items and how to verify each one.

---

## 1. Request history & “last request” system search

**What changed**
- **New file:** `agent/request_history.py` — stores last N requests (query, intent, success/fail, tool, error_type, summary).
- **System Search** in `agent/graph.py`: if the user asks about “last request” / “status of my request” / “what did you do”, the agent answers from request history instead of the tool registry.
- **UI:** `st.session_state.request_history` is kept and passed into `run_agent_full()`; the returned `request_history` is written back so it persists across turns.
- **Recording:** Every completed turn (API success/failure, clarification, system search, RAG) appends one entry to `request_history`.

**How to test**
1. Restart the app: `streamlit run ui/app.py` (from the `paypal-agent` folder).
2. Do at least one action, e.g. **“What tools are available for managing invoices?”** (system search) or **“Send an invoice for $50 to john@example.com”** (API).
3. In the next message ask: **“What was the status of my last request?”** or **“What did you do last?”**
4. **Expected:** You get a short list of recent requests (your last 1–5) with ✅/❌, query, intent, and tool name — not the generic “tool registry” search result.
5. **Edge case:** Ask **“What was the status of my last request?”** before doing anything in a new session. **Expected:** Message like “No previous requests in this session.”

---

## 2. Multi-step execution

**What changed**
- **New flow:** After a successful API call, the graph routes to `after_success_router` → if intent is **multi_step** → **multi_step_decide**; else → response_synthesizer.
- **multi_step_decide** in `agent/graph.py`: appends the current step result, then either asks the LLM “one more step or done?” or stops at `MAX_MULTI_STEPS` (default 5). If “done” or max steps → response_synthesizer; if “next step” → sets a new `execution_plan` and goes back to **tool_executor**.
- **Config:** Env `MAX_MULTI_STEPS=5` (optional in `.env`).

**How to test**
1. Ask something that may be classified as **multi_step**, e.g. **“Create an invoice for $30 to alice@test.com and then send it.”**
2. **Expected:** Intent may be `multi_step`; the agent runs the first API (create invoice), then the “multi_step_decide” logic runs. If the LLM says “one more step”, it runs send; then you get a summary of both steps.
3. **Check logs (terminal):** You should see multiple “⚡ Executing” lines and “📋 Plan” for more than one step when multi-step is used.
4. **Note:** If the intent classifier labels it as **api_action**, you’ll only get one step (current behavior). Multi-step only runs when intent is **multi_step**.

---

## 3. Structured error handling

**What changed**
- **Error classification:** `_classify_error_type(api_result)` in `agent/graph.py` sets `error_type`: `auth`, `validation`, `not_found`, `rate_limit`, `server_error`, or `unknown`.
- **tool_executor** sets `result["error_type"]` and `state["error_type"]`.
- **validator:** On **auth** error, route to **give_up** (no retry).
- **failure_response:** Different user-facing messages per `error_type` (auth, rate_limit, not_found, validation, generic).

**How to test**
1. **Auth error:** In `.env` set a wrong `PAYPAL_CLIENT_ID` or `PAYPAL_SECRET`, restart, then e.g. **“List my invoices.”**  
   **Expected:** After failure, message mentions **authentication** and suggests checking Client ID/Secret; no retries (you should not see multiple “🔁 Retrying” for that request).
2. **Validation-style error:** Use a request that triggers a 400/422 (e.g. invalid body if you can force one).  
   **Expected:** Message says the request was **rejected** (invalid parameters/body) and suggests rephrasing.
3. **Normal success:** Restore correct `.env`, run **“List my invoices.”**  
   **Expected:** No error; list of invoices or a normal API response.

---

## 4. Retry with re-retrieve and replan

**What changed**
- **Retry path:** `retry_node` now goes to **tool_retriever** (re-retrieve tools), not directly to **llm_planner**.
- **llm_planner:** When `retry_count > 0`, the prompt includes “PREVIOUS ATTEMPT FAILED”, `error_type`, and a short error snippet, and tells the LLM to choose a different tool or fix parameters.

**How to test**
1. Trigger a **retryable** failure (e.g. a wrong but non-auth parameter so the first call fails and retry runs).
2. **Check terminal:** You should see “🔁 Retrying…”, then “🔍 Retrieved … tools” again (re-retrieval), then a new “📋 Plan” (replan).
3. **Expected:** Second attempt can use a different tool or different parameters, not the exact same call again.

---

## 5. Configurable top_k and confidence fallback

**What changed**
- **Env (optional):** `TOOL_TOP_K` (default 5), `TOOL_CONFIDENCE_THRESHOLD` (default 0.3) in `.env` or `.env.example`.
- **tool_retriever:** Uses `retrieve_top_tools(query, top_k=TOOL_TOP_K)`. If the **best similarity score < threshold**, sets `skip_to_clarification`.
- **clarification_node:** Returns a message asking the user to rephrase or be more specific, then END.
- **Routing:** After tool_retriever, if `skip_to_clarification` → clarification_node → END; else → llm_planner.

**How to test**
1. **Clarification (low confidence):** Ask something very vague or off-topic, e.g. **“Do the thing with the stuff.”** or **“asdfghjkl”**.  
   **Expected:** Reply like “I’m not sure which action you mean. Could you rephrase or be more specific? For example: …” and no API call.
2. **Different threshold:** In `.env` set `TOOL_CONFIDENCE_THRESHOLD=0.9`, restart, then ask **“List my invoices.”**  
   **Expected:** If the best score is below 0.9, you get the clarification message instead of running the API. Revert to 0.3 (or remove the line) to restore normal behavior.
3. **top_k:** Set `TOOL_TOP_K=3` in `.env`, restart, run any API-style query. In logs you should see “Retrieved 3 tools” (or similar).

---

## 6. Conversation history in planner

**What changed**
- **llm_planner** in `agent/graph.py`: reads the last few entries from `state["messages"]`, formats them as “User: … / Assistant: …”, and adds a **“Recent conversation (for context)”** section to the planning prompt.
- When **retry_count > 0**, the planner also gets the “PREVIOUS ATTEMPT FAILED” block (see section 4).

**How to test**
1. Send two messages in sequence, e.g.  
   - “Create an invoice for $10 to bob@test.com”  
   - “Now send that invoice.”  
2. **Expected:** The second turn can use context from the first (e.g. invoice ID) if the planner and tools support it. You can also check in LangSmith that the planner prompt for the second turn contains “Recent conversation” with the first exchange.

---

## 7. Request history wired in UI and agent

**What changed**
- **Agent:** `_build_initial_state(user_query, request_history=None)` and `run_agent_full(user_query, request_history=None)` accept and return **request_history**.
- **UI:** Initializes `st.session_state.request_history = []` and passes it into `run_agent_full(prompt, request_history=st.session_state.request_history)`; after the call, updates `st.session_state.request_history = result["request_history"]`.
- **Recording:** Success/failure/clarification/system_search/RAG nodes append to `request_history` and return it in state.

**How to test**
1. Run 2–3 different actions (e.g. one system search, one “list invoices”, one “what was my last request?”).
2. **Expected:** “What was the status of my last request?” shows those 2–3 entries in order.
3. Refresh the page (new session). **Expected:** Request history is empty again (it’s per session, not persisted to disk).

---

## Quick checklist

| # | Feature | How to check |
|---|--------|---------------|
| 1 | Request history + “last request” | Ask “What was the status of my last request?” after doing 1–2 actions; see list of recent requests. |
| 2 | Multi-step | Ask “Create an invoice for $30 to x@y.com and send it”; see multiple steps in logs and in the final summary. |
| 3 | Structured errors | Wrong PayPal credentials → auth message, no retry. Invalid request → validation-style message. |
| 4 | Retry + replan | Force a retry; see “Retrying”, then “Retrieved … tools” and a new “Plan”. |
| 5 | Confidence fallback | Ask “asdfghjkl” or “Do the thing” → clarification message, no API call. |
| 6 | Conversation in planner | Two-turn flow (“create invoice” then “send it”); second turn uses context. |
| 7 | Request history in UI | Same as #1; history persists across turns in the same session. |

---

## Env vars (optional)

In `.env` you can add (defaults are used if omitted):

```env
TOOL_TOP_K=5
TOOL_CONFIDENCE_THRESHOLD=0.3
MAX_MULTI_STEPS=5
REQUEST_HISTORY_MAX=20
```

---

## Files that were added or modified

- **New:** `agent/request_history.py`
- **New:** `docs/CHANGES_AND_TESTING.md` (this file)
- **Modified:** `agent/graph.py` (state, nodes, routing, retry, multi_step, errors, history, config)
- **Modified:** `ui/app.py` (request_history in session, one new example query)
- **Modified:** `.env.example` (optional agent config vars)

After changing code or `.env`, restart Streamlit: stop the app (Ctrl+C) and run `streamlit run ui/app.py` again from the `paypal-agent` directory.
