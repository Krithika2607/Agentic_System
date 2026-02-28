# 💳 PayPal AI Agent

A scalable agentic system that lets users interact with 100+ PayPal APIs through natural language chat.

## Architecture

```
User Query
    ↓
Intent Router (LLM classifies: api_action / knowledge / system_search)
    ↓
Semantic Tool Retriever (ChromaDB → top 5 tools from 100+)
    ↓
LLM Planner (LLaMA 3.1 70B via Groq — sees only top 5 tools)
    ↓
Tool Executor (hits real PayPal Sandbox API)
    ↓
Validator → Response Synthesizer
    ↓
User Response
```

## Tech Stack

| Component | Technology |
|---|---|
| LLM | LLaMA 3.1 70B (via Groq) |
| Embeddings | BAAI/bge-small-en-v1.5 (open-source) |
| Vector DB | ChromaDB (local, persistent) |
| Agent Framework | LangGraph |
| Observability | LangSmith |
| UI | Streamlit |
| APIs | PayPal Sandbox |

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env with your credentials
```

Required credentials:
- `GROQ_API_KEY` — from https://console.groq.com
- `PAYPAL_CLIENT_ID` — from https://developer.paypal.com → Apps & Credentials → Sandbox
- `PAYPAL_SECRET` — same as above
- `LANGCHAIN_API_KEY` — from https://smith.langchain.com (free)

### 3. Add your Postman collection
```bash
# Export PayPal Postman collection as JSON
# Place it at:
data/collection.json
```

### 4. Run ingestion (ONE TIME ONLY)
```bash
python ingest.py --collection data/collection.json
```

This will:
- Parse all 100+ APIs from the collection
- Embed each tool as a single vector (no chunking)
- Store in ChromaDB locally

### 5. Launch the app
```bash
streamlit run ui/app.py
```

## Example Queries

```
"Send an invoice for $50 to john@example.com"
"What was my total sales volume last month?"
"Is there a dispute open from user_123?"
"Create an order for $100 USD"
"What tools are available for managing invoices?"
"How does PayPal handle disputes?"
```

## Project Structure

```
paypal-agent/
├── data/
│   └── collection.json          ← Your Postman export (add this)
├── tools/
│   ├── parse_collection.py      ← Parses Postman JSON → tool definitions
│   └── paypal_executor.py       ← Handles PayPal auth + API calls
├── vectorstore/
│   └── embed_tools.py           ← Embeds tools + ChromaDB operations
├── agent/
│   └── graph.py                 ← Full LangGraph orchestration
├── ui/
│   └── app.py                   ← Streamlit chat interface
├── ingest.py                    ← Run once to ingest tools
├── requirements.txt
├── .env.example
└── README.md
```

## Key Design Decisions

**Why no chunking?** Each tool definition is small (~10 lines). Chunking would split the endpoint from its parameters, breaking the system. One tool = one vector.

**Why BAAI/bge-small-en-v1.5?** Best open-source embedding model for semantic similarity. Runs locally, no API key needed, excellent performance.

**Why LangGraph over LangChain?** LangGraph handles stateful multi-step flows, retry loops, and conditional routing — essential for a production agent.

**Why Groq + LLaMA 3.1 70B?** Fastest inference for a 70B model. Free tier available. Comparable to GPT-4 for structured tasks.
