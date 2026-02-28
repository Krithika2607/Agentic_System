"""
agent/rag_agent.py

RAG Pipeline — answers questions from uploaded documents.
Uses same LLM (LLaMA 3.1 70B via Groq) and embedding model.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from vectorstore.rag_store import retrieve_relevant_chunks, list_documents

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


def answer_from_documents(query: str, filename_filter: str = None) -> dict:
    """
    Answer a user question using the RAG pipeline.

    Steps:
    1. Retrieve top-k relevant chunks from knowledge base
    2. Build context from chunks
    3. LLM generates grounded answer
    4. Return answer + sources

    Args:
        query: User's question
        filename_filter: Optionally restrict to one document

    Returns:
        dict with answer, sources, chunks_used
    """

    # Step 1: Check documents exist
    docs = list_documents()
    if not docs:
        return {
            "answer": "⚠️ No documents uploaded yet. Please upload a PDF or text file first using the upload button above.",
            "sources": [],
            "chunks_used": 0
        }

    # Step 2: Retrieve relevant chunks
    chunks = retrieve_relevant_chunks(
        query=query,
        top_k=5,
        filename_filter=filename_filter
    )

    if not chunks:
        return {
            "answer": "I couldn't find relevant information in the uploaded documents for your question.",
            "sources": [],
            "chunks_used": 0
        }

    # Step 3: Build context
    context = ""
    sources = []
    for i, chunk in enumerate(chunks):
        context += f"\n--- Source {i+1} (from {chunk['metadata']['filename']}, relevance: {chunk['similarity']}) ---\n"
        context += chunk["text"] + "\n"
        src = chunk["metadata"]["filename"]
        if src not in sources:
            sources.append(src)

    # Step 4: LLM answers grounded in context
    prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided document context.

Document Context:
{context}

User Question: {query}

Instructions:
- Answer based ONLY on the context provided above
- If the answer is not in the context, say "I couldn't find this in the uploaded documents"
- Be specific and cite which part of the document supports your answer
- Use markdown formatting for clarity
- Keep the answer concise but complete"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "answer": response.content,
        "sources": sources,
        "chunks_used": len(chunks),
        "chunks": chunks  # for debug/display
    }