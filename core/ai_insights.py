"""
Sends parsed-genome KPIs/summary stats to an LLM for a plain-language
interpretation. Supports Gemini (as in the original spec) and Anthropic's
Claude as an alternative provider — pick whichever key you have.

Both providers are called via plain HTTPS requests, no SDK required, so
there's nothing extra to install beyond `requests`.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

SYSTEM_INSTRUCTION = (
    "You are a genomics analyst. Analyze the given genomic KPIs and "
    "sequence motifs and provide a plain-language summary of the "
    "biological significance, focusing on potential pathogenic "
    "anomalies or regulatory features. Be concise and avoid overclaiming "
    "clinical significance from summary statistics alone."
)


def _create_session_with_retries(max_retries: int = 3, backoff_factor: float = 1.0) -> requests.Session:
    """Create a requests session with exponential backoff retry strategy."""
    session = requests.Session()
    
    # Configure retry strategy for 429 (Too Many Requests) and other transient errors
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
        allowed_methods=["POST", "GET"],
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


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

    prompt = _build_prompt(kpi_summary, extra_context)
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "contents": [{"parts": [{"text": prompt}]}],
    }
    
    session = _create_session_with_retries(max_retries=5, backoff_factor=2.0)
    max_attempts = 5
    attempt = 0
    
    while attempt < max_attempts:
        try:
            resp = session.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                # Extract retry-after header if available
                retry_after = e.response.headers.get("Retry-After")
                if retry_after:
                    wait_time = int(retry_after)
                else:
                    # Exponential backoff: 2^attempt seconds
                    wait_time = min(2 ** attempt, 32)
                
                attempt += 1
                if attempt < max_attempts:
                    print(f"Rate limited. Waiting {wait_time}s before retry {attempt}/{max_attempts}...")
                    time.sleep(wait_time)
                    continue
            raise
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to get AI insights from Gemini: {str(e)}")
    
    raise RuntimeError(f"Failed to get AI insights after {max_attempts} attempts")


def get_insights_claude(kpi_summary: dict, extra_context: str = "", api_key: Optional[str] = None) -> str:
    api_key = _get_secret_value("ANTHROPIC_API_KEY", api_key)
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

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
    
    session = _create_session_with_retries(max_retries=5, backoff_factor=2.0)
    max_attempts = 5
    attempt = 0
    
    while attempt < max_attempts:
        try:
            resp = session.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return "".join(block["text"] for block in data["content"] if block["type"] == "text")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                # Extract retry-after header if available
                retry_after = e.response.headers.get("Retry-After")
                if retry_after:
                    wait_time = int(retry_after)
                else:
                    # Exponential backoff: 2^attempt seconds
                    wait_time = min(2 ** attempt, 32)
                
                attempt += 1
                if attempt < max_attempts:
                    print(f"Rate limited. Waiting {wait_time}s before retry {attempt}/{max_attempts}...")
                    time.sleep(wait_time)
                    continue
            raise
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to get AI insights from Claude: {str(e)}")
    
    raise RuntimeError(f"Failed to get AI insights after {max_attempts} attempts")


def get_insights(kpi_summary: dict, extra_context: str = "", provider: str = "gemini", api_key: Optional[str] = None) -> str:
    if provider == "gemini":
        return get_insights_gemini(kpi_summary, extra_context, api_key=api_key)
    if provider == "claude":
        return get_insights_claude(kpi_summary, extra_context, api_key=api_key)
    raise ValueError(f"Unknown provider: {provider}")
