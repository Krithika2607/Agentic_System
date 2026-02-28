"""
vectorstore/rag_store.py

Handles document ingestion and retrieval for the RAG Pipeline Tool.
- Chunks uploaded documents
- Embeds using same mxbai-embed-large-v1 model
- Stores in separate ChromaDB collection: "knowledge_base"
- Retrieves relevant chunks for user questions
"""

import os
import json
import uuid
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

load_dotenv()

# Resolve Chroma path from project root so it matches ingest.py / embed_tools regardless of cwd
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_chroma_dir_env = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore/chroma_db")
CHROMA_DIR = _chroma_dir_env if os.path.isabs(_chroma_dir_env) else os.path.normpath(os.path.join(_project_root, _chroma_dir_env))
CHROMA_DIR = os.path.abspath(CHROMA_DIR)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mixedbread-ai/mxbai-embed-large-v1")
RAG_COLLECTION = "knowledge_base"

# Reuse same cached model as embed_tools.py
_model_cache = None


def get_embedding_model():
    global _model_cache
    if _model_cache is None:
        print(f"🔄 Loading embedding model: {EMBEDDING_MODEL}")
        _model_cache = SentenceTransformer(EMBEDDING_MODEL)
        print(f"✅ Embedding model loaded")
    return _model_cache


def get_chroma_client():
    return chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )


def get_or_create_rag_collection():
    client = get_chroma_client()
    try:
        return client.get_collection(RAG_COLLECTION)
    except Exception:
        return client.create_collection(
            name=RAG_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )


# ─────────────────────────────────────────
# Document Chunking
# ─────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks.
    Unlike tools (no chunking), documents ARE chunked
    because they can be very large.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap  # overlap for context continuity

    return chunks


# ─────────────────────────────────────────
# Ingest a Document
# ─────────────────────────────────────────
def ingest_document(text: str, filename: str, doc_type: str = "pdf") -> dict:
    """
    Ingest a document into the RAG knowledge base.

    Args:
        text: Full extracted text of the document
        filename: Original filename (for metadata)
        doc_type: Type of document (pdf, txt, md)

    Returns:
        Summary dict with chunk count
    """
    model = get_embedding_model()
    collection = get_or_create_rag_collection()

    # Chunk the document
    chunks = chunk_text(text, chunk_size=150, overlap=20)
    print(f"📄 Ingesting '{filename}': {len(chunks)} chunks")

    # Generate unique doc ID
    doc_id = str(uuid.uuid4())[:8]

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{i}"
        embedding = model.encode(chunk).tolist()

        ids.append(chunk_id)
        embeddings.append(embedding)
        documents.append(chunk)
        metadatas.append({
            "filename": filename,
            "doc_id": doc_id,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "doc_type": doc_type
        })

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    print(f"✅ Stored {len(chunks)} chunks from '{filename}'")
    return {
        "filename": filename,
        "doc_id": doc_id,
        "total_chunks": len(chunks),
        "success": True
    }


# ─────────────────────────────────────────
# Retrieve Relevant Chunks
# ─────────────────────────────────────────
def retrieve_relevant_chunks(query: str, top_k: int = 5, filename_filter: str = None) -> list[dict]:
    """
    Semantic search over knowledge base chunks.

    Args:
        query: User's question
        top_k: Number of chunks to return
        filename_filter: Optionally filter by specific document

    Returns:
        List of relevant chunks with metadata
    """
    model = get_embedding_model()
    collection = get_or_create_rag_collection()

    # Check if collection has documents
    if collection.count() == 0:
        return []

    query_embedding = model.encode(query).tolist()

    where_filter = None
    if filename_filter:
        where_filter = {"filename": filename_filter}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["metadatas", "distances", "documents"],
        where=where_filter
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "similarity": round(1 - results["distances"][0][i], 3)
        })

    return chunks


# ─────────────────────────────────────────
# List All Ingested Documents
# ─────────────────────────────────────────
def list_documents() -> list[dict]:
    """Return a list of all unique documents in the knowledge base."""
    collection = get_or_create_rag_collection()

    if collection.count() == 0:
        return []

    results = collection.get(include=["metadatas"])
    seen = {}

    for meta in results["metadatas"]:
        doc_id = meta.get("doc_id", "unknown")
        if doc_id not in seen:
            seen[doc_id] = {
                "doc_id": doc_id,
                "filename": meta.get("filename", "unknown"),
                "total_chunks": meta.get("total_chunks", 0),
                "doc_type": meta.get("doc_type", "unknown")
            }

    return list(seen.values())


# ─────────────────────────────────────────
# Delete a Document
# ─────────────────────────────────────────
def delete_document(doc_id: str):
    """Remove all chunks of a specific document."""
    collection = get_or_create_rag_collection()
    collection.delete(where={"doc_id": doc_id})
    print(f"🗑️ Deleted document: {doc_id}")


# ─────────────────────────────────────────
# Groq Vision: describe image for RAG (free tier)
# ─────────────────────────────────────────
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_IMAGES_PER_PDF = 15
MAX_IMAGE_BYTES_FOR_GROQ = 3 * 1024 * 1024  # 3MB (Groq base64 limit 4MB)


def _describe_image_with_groq(image_bytes: bytes, mime_type: str = "image/png") -> str:
    """Use Groq's free vision model to describe an image. Returns text for RAG."""
    import base64
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return ""
    if len(image_bytes) > MAX_IMAGE_BYTES_FOR_GROQ:
        return "[Image too large for vision model]"
    try:
        from groq import Groq
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        url = f"data:{mime_type};base64,{b64}"
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in detail for a document knowledge base. Include any text, numbers, labels, diagrams, tables, or structure visible. Be concise but complete so someone can answer questions from your description."
                        },
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                }
            ],
            max_tokens=512,
            temperature=0.2,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"⚠️ Groq vision failed: {e}")
        return ""


def _extract_images_from_pdf(pdf_bytes: bytes) -> list[tuple[bytes, str, int]]:
    """Extract embedded images from PDF. Returns list of (image_bytes, mime_type, page_1based)."""
    import io
    out = []
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_index in range(len(doc)):
            page = doc[page_index]
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    base = doc.extract_image(xref)
                    img_bytes = base["image"]
                    ext = (base.get("ext") or "png").lower()
                    mime = "image/png" if ext == "png" else "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
                    out.append((img_bytes, mime, page_index + 1))
                except Exception:
                    continue
        doc.close()
    except ImportError:
        pass
    except Exception:
        pass
    return out


# ─────────────────────────────────────────
# Extract text from PDF bytes (text + tables + image descriptions via Groq Vision)
# ─────────────────────────────────────────
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text, tables, and image descriptions from PDF for RAG.
    - Body text: pypdf
    - Tables: pdfplumber → markdown-like text
    - Images: PyMuPDF extract → Groq Vision (Llama 4 Scout) → description text
    """
    import io
    parts = []

    # 1) Body text via pypdf
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            t = page.extract_text()
            if t and t.strip():
                parts.append(t.strip())
    except ImportError:
        pass
    except Exception:
        pass

    # 2) Tables via pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        if not table:
                            continue
                        lines = []
                        header = table[0]
                        lines.append(" | ".join(str(c) if c else "" for c in header))
                        lines.append(" | ".join("---" for _ in header))
                        for row in table[1:]:
                            lines.append(" | ".join(str(c) if c else "" for c in row))
                        parts.append("Table (page {}):\n".format(i + 1) + "\n".join(lines))
    except ImportError:
        pass
    except Exception:
        pass

    # 3) Images → Groq Vision descriptions (skip if DISABLE_PDF_VISION=1 for faster ingest)
    if os.getenv("DISABLE_PDF_VISION", "").lower() not in ("1", "true", "yes"):
        images = _extract_images_from_pdf(pdf_bytes)
        for i, (img_bytes, mime, page_no) in enumerate(images[:MAX_IMAGES_PER_PDF]):
            desc = _describe_image_with_groq(img_bytes, mime)
            if desc:
                parts.append("Image (page {}):\n{}".format(page_no, desc))

    if not parts:
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return (text + "\n").strip()
        except Exception as e:
            raise Exception(f"PDF extraction failed: {e}")

    return "\n\n".join(parts)