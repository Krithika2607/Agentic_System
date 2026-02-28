"""
tools/paypal_executor.py

Handles PayPal Sandbox authentication and executes API calls.
"""

import os
import json
import requests
import base64
from dotenv import load_dotenv

load_dotenv()

PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID")
PAYPAL_SECRET = os.getenv("PAYPAL_SECRET")
PAYPAL_BASE_URL = os.getenv("PAYPAL_BASE_URL", "https://api-m.sandbox.paypal.com")

_access_token = None


def get_access_token() -> str:
    global _access_token
    if _access_token:
        return _access_token

    credentials = f"{PAYPAL_CLIENT_ID}:{PAYPAL_SECRET}"
    encoded = base64.b64encode(credentials.encode()).decode()

    response = requests.post(
        f"{PAYPAL_BASE_URL}/v1/oauth2/token",
        headers={
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/x-www-form-urlencoded"
        },
        data="grant_type=client_credentials"
    )

    if response.status_code == 200:
        _access_token = response.json()["access_token"]
        return _access_token
    else:
        raise Exception(f"PayPal auth failed: {response.status_code} {response.text}")


def execute_paypal_api(method: str, endpoint: str, body: dict = None, params: dict = None) -> dict:
    token = get_access_token()

    if endpoint.startswith("http"):
        url = endpoint
    else:
        url = f"{PAYPAL_BASE_URL}{endpoint}"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # ── DEBUG: print exact body being sent
    print(f"📦 Sending body: {json.dumps(body, indent=2) if body else 'None'}")

    response = requests.request(
        method=method.upper(),
        url=url,
        headers=headers,
        json=body,
        params=params
    )

    # ── DEBUG: print error response
    if not response.ok:
        print(f"🔴 PayPal error {response.status_code}: {response.text[:800]}")

    result = {
        "status_code": response.status_code,
        "success": response.status_code in [200, 201, 202, 204],
        "data": {}
    }

    try:
        result["data"] = response.json()
    except Exception:
        result["data"] = {"message": response.text}

    return result


def test_connection() -> bool:
    try:
        token = get_access_token()
        return bool(token)
    except Exception as e:
        print(f"❌ PayPal connection failed: {e}")
        return False