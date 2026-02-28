"""
ingest.py

Run this ONCE to:
1. Parse the PayPal Postman collection JSON
2. Save parsed tools to data/parsed_tools.json  ← SEE THIS FILE!
3. Embed all tools using mxbai-embed-large-v1
4. Store in ChromaDB

Usage:
    python ingest.py --collection data/collection.json
"""

import argparse
import sys
import os
import json

def main():
    parser = argparse.ArgumentParser(description="Ingest PayPal Postman collection into ChromaDB")
    parser.add_argument(
        "--collection",
        default="data/collection.json",
        help="Path to Postman collection JSON file"
    )
    parser.add_argument(
        "--parsed-output",
        default="data/parsed_tools.json",
        help="Where to save the parsed tools JSON (so you can inspect them)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.collection):
        print(f"❌ Collection file not found: {args.collection}")
        print(f"   Please place your exported Postman JSON at: {args.collection}")
        sys.exit(1)

    print("=" * 60)
    print("🚀 PayPal Agent — Tool Ingestion Pipeline")
    print("=" * 60)

    # ─────────────────────────────────────────
    # STEP 1: Parse collection
    # ─────────────────────────────────────────
    print("\n📂 Step 1: Parsing Postman collection...")
    from tools.parse_collection import parse_collection
    tools = parse_collection(args.collection)

    if not tools:
        print("❌ No tools found in collection. Check your JSON file.")
        sys.exit(1)

    from collections import Counter
    category_counts = Counter(t["folder"] for t in tools)
    print(f"\n   ✅ Found {len(tools)} tools across {len(category_counts)} categories:")
    for category, count in sorted(category_counts.items()):
        print(f"      • {category}: {count} tools")

    # ─────────────────────────────────────────
    # STEP 2: Save parsed tools to JSON file
    # so you can OPEN and INSPECT the structure
    # ─────────────────────────────────────────
    print(f"\n💾 Step 2: Saving parsed tools to → {args.parsed_output}")

    os.makedirs(os.path.dirname(args.parsed_output), exist_ok=True)

    # Save full structure
    output = {
        "total_tools": len(tools),
        "categories": dict(category_counts),
        "tools": tools
    }

    with open(args.parsed_output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"   ✅ Saved! Open this file to see all tool definitions:")
    print(f"      📄 {args.parsed_output}")
    print(f"\n   Here's a preview of the first tool:")
    print(f"   {json.dumps(tools[0], indent=4)[:600]}...")

    # ─────────────────────────────────────────
    # STEP 3: Embed and store in ChromaDB
    # ─────────────────────────────────────────
    print(f"\n🔢 Step 3: Embedding {len(tools)} tools with mxbai-embed-large-v1...")
    print(f"   (One vector per tool — NO chunking)")
    from vectorstore.embed_tools import embed_and_store_tools
    embed_and_store_tools(tools)

    # ─────────────────────────────────────────
    # STEP 4: Test retrieval
    # ─────────────────────────────────────────
    print(f"\n🧪 Step 4: Testing semantic retrieval...")
    from vectorstore.embed_tools import retrieve_top_tools

    test_queries = [
        "send invoice to customer",
        "check payment status",
        "handle dispute",
        "create subscription"
    ]

    for query in test_queries:
        results = retrieve_top_tools(query, top_k=3)
        print(f"\n   Query: '{query}'")
        for r in results:
            print(f"   → {r['name']} | {r['folder']} (score: {r['similarity_score']})")

    print("\n" + "=" * 60)
    print("✅ Ingestion complete!")
    print(f"\n   📄 Inspect parsed tools : {args.parsed_output}")
    print(f"   🗄️  ChromaDB stored at   : {os.getenv('CHROMA_PERSIST_DIR', './vectorstore/chroma_db')}")
    print(f"\n   ▶️  Run the app          : streamlit run ui/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()