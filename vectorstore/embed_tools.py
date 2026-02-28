"""
vectorstore/embed_tools.py

Embeds each tool definition using BAAI/bge-small-en-v1.5
(best open-source embedding model) and stores in ChromaDB.
One tool = One vector. NO chunking.
"""

import os
import json
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

load_dotenv()

# Resolve Chroma path from project root (parent of vectorstore/) so it's correct regardless of cwd
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_chroma_dir_env = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore/chroma_db")
CHROMA_DIR = _chroma_dir_env if os.path.isabs(_chroma_dir_env) else os.path.normpath(os.path.join(_project_root, _chroma_dir_env))
CHROMA_DIR = os.path.abspath(CHROMA_DIR)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mixedbread-ai/mxbai-embed-large-v1")
COLLECTION_NAME = "paypal_tools"


_model_cache = None

def get_embedding_model():
    """Load the open-source embedding model. Cached after first load."""
    global _model_cache
    if _model_cache is None:
        print(f"🔄 Loading embedding model: {EMBEDDING_MODEL}")
        _model_cache = SentenceTransformer(EMBEDDING_MODEL)
        print(f"✅ Embedding model loaded")
    return _model_cache


def get_chroma_client():
    """Get a persistent ChromaDB client."""
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    return client


def embed_and_store_tools(tools: list[dict]):
    """
    Takes parsed tool definitions, embeds each one as a SINGLE vector
    (no chunking), and stores in ChromaDB with full metadata.
    """
    model = get_embedding_model()
    client = get_chroma_client()

    # Delete existing collection if rebuilding
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"🗑️  Cleared existing collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"🔄 Embedding {len(tools)} tools...")

    ids = []
    embeddings = []
    metadatas = []
    documents = []

    for i, tool in enumerate(tools):
        # Embed the FULL text as ONE vector — no chunking
        text_to_embed = tool["full_text"]
        embedding = model.encode(text_to_embed).tolist()

        tool_id = f"tool_{i}_{tool['tool_id'][:50]}"

        ids.append(tool_id)
        embeddings.append(embedding)
        documents.append(text_to_embed)
        metadatas.append({
            "name": tool["name"],
            "folder": tool["folder"],
            "method": tool["method"],
            "endpoint": tool["endpoint"],
            "description": tool["description"],
            "parameters": json.dumps(tool["parameters"]),
            "tool_id": tool["tool_id"]
        })

    # Store all at once
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    print(f"✅ Successfully stored {len(tools)} tools in ChromaDB at {CHROMA_DIR}")
    return collection


def retrieve_top_tools(query: str, top_k: int = 5) -> list[dict]:
    """
    Semantic search: embed user query and find top-k most relevant tools.
    This is the core of the system — LLM only sees these top-k tools.
    """
    model = get_embedding_model()
    client = get_chroma_client()
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "distances", "documents"]
    )

    tools = []
    for i, metadata in enumerate(results["metadatas"][0]):
        tool = {
            **metadata,
            "parameters": json.loads(metadata.get("parameters", "{}")),
            "similarity_score": round(1 - results["distances"][0][i], 3)
        }
        tools.append(tool)

    return tools


def search_tool_registry(query: str) -> str:
    """System Search Tool — searches tool registry for capabilities."""
    tools = retrieve_top_tools(query, top_k=8)
    result = f"Found {len(tools)} relevant tools for '{query}':\n\n"
    for t in tools:
        result += f"• **{t['name']}** ({t['folder']}) — {t['method']} {t['endpoint']}\n"
        result += f"  Score: {t['similarity_score']}\n\n"
    return result


if __name__ == "__main__":
    from tools.parse_collection import parse_collection
    tools = parse_collection("data/collection.json")
    embed_and_store_tools(tools)

    # Test retrieval
    results = retrieve_top_tools("send invoice to customer")
    print("\n🔍 Test query: 'send invoice to customer'")
    for r in results:
        print(f"  → {r['name']} (score: {r['similarity_score']})")