"""
Sends parsed-genome KPIs/summary stats to an LLM for a plain-language
interpretation. Supports Gemini (as in the original spec) and Anthropic's
Claude as an alternative provider — pick whichever key you have.

Both providers are called via plain HTTPS requests, no SDK required, so
there's nothing extra to install beyond `requests`.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Optional

import requests

SYSTEM_INSTRUCTION = (
    "You are a genomics analyst. Analyze the given genomic KPIs and "
    "sequence motifs and provide a plain-language summary of the "
    "biological significance, focusing on potential pathogenic "
    "anomalies or regulatory features. Be concise and avoid overclaiming "
    "clinical significance from summary statistics alone."
)

# Cache for AI insights to reduce API calls
_insights_cache: dict[str, tuple[float, str]] = {}
_last_request_time: dict[str, float] = {}
MIN_REQUEST_INTERVAL = 2  # Minimum seconds between requests per provider


def _get_cache_key(kpi_summary: dict, extra_context: str, provider: str) -> str:
    """Generate a cache key from KPI data and provider."""
    data = f"{provider}|{str(sorted(kpi_summary.items()))}|{extra_context}"
    return hashlib.md5(data.encode()).hexdigest()


def _get_cached_insight(cache_key: str, max_age: int = 3600) -> Optional[str]:
    """Retrieve cached insight if it exists and hasn't expired."""
    if cache_key in _insights_cache:
        timestamp, result = _insights_cache[cache_key]
        if time.time() - timestamp < max_age:
            print(f"Using cached AI insight (age: {int(time.time() - timestamp)}s)")
            return result
        else:
            del _insights_cache[cache_key]
    return None


def _cache_insight(cache_key: str, result: str) -> None:
    """Store insight in cache."""
    _insights_cache[cache_key] = (time.time(), result)


def _throttle_request(provider: str) -> None:
    """Enforce minimum time between requests to the same provider."""
    if provider in _last_request_time:
        elapsed = time.time() - _last_request_time[provider]
        if elapsed < MIN_REQUEST_INTERVAL:
            wait_time = MIN_REQUEST_INTERVAL - elapsed
            print(f"Throttling {provider} requests. Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
    
    _last_request_time[provider] = time.time()


def _build_prompt(kpi_summary: dict, extra_context: str = "") -> str:
    lines = [f"{k}: {v}" for k, v in kpi_summary.items()]
    prompt = "Genomic KPIs:\n" + "\n".join(lines)
    if extra_context:
        prompt += f"\n\nAdditional context:\n{extra_context}"
    return prompt


def _get_secret_value(name: str, api_key: Optional[str] = None) -> Optional[str]:
    if api_key:
        return api_key

    try:
        import streamlit as st

        secret_value = st.secrets.get(name)
        if secret_value:
            return str(secret_value)
    except Exception:
        pass

    return os.environ.get(name)


def get_insights_gemini(kpi_summary: dict, extra_context: str = "", api_key: Optional[str] = None) -> str:
    api_key = _get_secret_value("GEMINI_API_KEY", api_key)
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    # Check cache first
    cache_key = _get_cache_key(kpi_summary, extra_context, "gemini")
    cached_result = _get_cached_insight(cache_key)
    if cached_result:
        return cached_result

    # Throttle requests to avoid rate limiting
    _throttle_request("gemini")

    prompt = _build_prompt(kpi_summary, extra_context)
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "contents": [{"parts": [{"text": prompt}]}],
    }
    
    max_attempts = 3
    base_wait = 5  # Start with 5 second wait for rate limiting
    
    for attempt in range(max_attempts):
        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            result = data["candidates"][0]["content"]["parts"][0]["text"]
            _cache_insight(cache_key, result)
            return result
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                if attempt < max_attempts - 1:
                    # Check for Retry-After header
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after:
                        wait_time = int(retry_after)
                    else:
                        # Exponential backoff: 5, 10, 20 seconds
                        wait_time = base_wait * (2 ** attempt)
                    
                    print(f"Rate limited (429). Waiting {wait_time}s before retry {attempt + 1}/{max_attempts}...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(
                        f"Gemini API rate limit exceeded. Your API key may have reached its quota. "
                        f"Please check your API usage at https://console.cloud.google.com or wait before retrying."
                    )
            raise RuntimeError(f"Gemini API error: {e.response.status_code} - {e.response.text}")
        except requests.exceptions.Timeout:
            raise RuntimeError("Gemini API request timed out (30s)")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to connect to Gemini API: {str(e)}")
    
    raise RuntimeError(f"Failed to get AI insights after {max_attempts} attempts")


def get_insights_claude(kpi_summary: dict, extra_context: str = "", api_key: Optional[str] = None) -> str:
    api_key = _get_secret_value("ANTHROPIC_API_KEY", api_key)
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    # Check cache first
    cache_key = _get_cache_key(kpi_summary, extra_context, "claude")
    cached_result = _get_cached_insight(cache_key)
    if cached_result:
        return cached_result

    # Throttle requests to avoid rate limiting
    _throttle_request("claude")

    prompt = _build_prompt(kpi_summary, extra_context)
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 500,
        "system": SYSTEM_INSTRUCTION,
        "messages": [{"role": "user", "content": prompt}],
    }
    
    max_attempts = 3
    base_wait = 5  # Start with 5 second wait for rate limiting
    
    for attempt in range(max_attempts):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            result = "".join(block["text"] for block in data["content"] if block["type"] == "text")
            _cache_insight(cache_key, result)
            return result
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                if attempt < max_attempts - 1:
                    # Check for Retry-After header
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after:
                        wait_time = int(retry_after)
                    else:
                        # Exponential backoff: 5, 10, 20 seconds
                        wait_time = base_wait * (2 ** attempt)
                    
                    print(f"Rate limited (429). Waiting {wait_time}s before retry {attempt + 1}/{max_attempts}...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(
                        f"Claude API rate limit exceeded. Your API key may have reached its quota. "
                        f"Please check your account usage or wait before retrying."
                    )
            raise RuntimeError(f"Claude API error: {e.response.status_code} - {e.response.text}")
        except requests.exceptions.Timeout:
            raise RuntimeError("Claude API request timed out (30s)")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to connect to Claude API: {str(e)}")
    
    raise RuntimeError(f"Failed to get AI insights after {max_attempts} attempts")


def get_insights(kpi_summary: dict, extra_context: str = "", provider: str = "gemini", api_key: Optional[str] = None) -> str:
    if provider == "gemini":
        return get_insights_gemini(kpi_summary, extra_context, api_key=api_key)
    if provider == "claude":
        return get_insights_claude(kpi_summary, extra_context, api_key=api_key)
    raise ValueError(f"Unknown provider: {provider}")
