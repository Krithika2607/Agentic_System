"""
tools/parse_collection.py

Parses PayPal Postman collection JSON and converts
each API into a structured tool definition.
"""

import json
import os
from typing import Any


def extract_params(request: dict) -> dict:
    """Extract parameters from a Postman request."""
    params = {}

    # Query params
    if request.get("url") and isinstance(request["url"], dict):
        for qp in request["url"].get("query", []):
            if not qp.get("disabled"):
                params[qp.get("key", "")] = {
                    "in": "query",
                    "description": qp.get("description", ""),
                    "value": qp.get("value", "")
                }

    # Body params
    body = request.get("body", {})
    if body:
        mode = body.get("mode", "")
        if mode == "raw":
            raw = body.get("raw", "")
            try:
                body_json = json.loads(raw)
                for key, val in body_json.items():
                    params[key] = {
                        "in": "body",
                        "description": f"Body parameter: {key}",
                        "value": val
                    }
            except Exception:
                params["raw_body"] = {
                    "in": "body",
                    "description": "Raw request body",
                    "value": raw[:500]
                }
        elif mode == "urlencoded":
            for item in body.get("urlencoded", []):
                params[item.get("key", "")] = {
                    "in": "body",
                    "description": item.get("description", ""),
                    "value": item.get("value", "")
                }
    return params


def extract_url(url_field: Any) -> str:
    """Extract clean URL string from Postman url field."""
    if isinstance(url_field, str):
        return url_field
    elif isinstance(url_field, dict):
        raw = url_field.get("raw", "")
        return raw.replace("{{base_url}}", "https://api-m.sandbox.paypal.com")
    return ""


def flatten_items(items: list, folder_name: str = "") -> list:
    """Recursively flatten nested Postman folders into a flat list of requests."""
    flat = []
    for item in items:
        if "item" in item:
            # It's a folder — recurse
            folder = item.get("name", "")
            flat.extend(flatten_items(item["item"], folder_name=folder))
        elif "request" in item:
            item["_folder"] = folder_name
            flat.append(item)
    return flat


def generate_description(name: str, method: str, endpoint: str, folder: str) -> str:
    """Generate a rich natural language description for semantic search."""
    return (
        f"Tool to {name.lower()}. "
        f"Category: {folder}. "
        f"HTTP Method: {method}. "
        f"Endpoint: {endpoint}. "
        f"Use this tool when the user wants to {name.lower()} "
        f"or perform {folder.lower()} related operations."
    )


def parse_collection(collection_path: str) -> list[dict]:
    """
    Parse a Postman collection JSON file and return
    a list of structured tool definitions.
    """
    with open(collection_path, "r", encoding="utf-8") as f:
        collection = json.load(f)

    # Handle both collection wrapper formats
    if "collection" in collection:
        items = collection["collection"].get("item", [])
    else:
        items = collection.get("item", [])

    flat_items = flatten_items(items)
    tools = []

    for item in flat_items:
        try:
            request = item.get("request", {})
            name = item.get("name", "unknown")
            method = request.get("method", "GET")
            url = extract_url(request.get("url", ""))
            params = extract_params(request)
            folder = item.get("_folder", "General")

            # Skip error examples (400, 401, 422 etc.)
            # Keep only successful API definitions
            if any(code in name for code in ["400", "401", "422", "404", "500"]):
                continue

            description = generate_description(name, method, url, folder)

            tool_def = {
                "tool_id": f"{folder.lower().replace(' ', '_')}_{name.lower().replace(' ', '_')[:40]}",
                "name": name,
                "folder": folder,
                "method": method,
                "endpoint": url,
                "description": description,
                "parameters": params,
                "full_text": f"{name} {folder} {method} {url} {description}"
            }
            tools.append(tool_def)

        except Exception as e:
            print(f"⚠️  Skipping item '{item.get('name', '?')}': {e}")
            continue

    print(f"✅ Parsed {len(tools)} tools from Postman collection")
    return tools


if __name__ == "__main__":
    tools = parse_collection("data/collection.json")
    for t in tools[:3]:
        print(json.dumps(t, indent=2))
