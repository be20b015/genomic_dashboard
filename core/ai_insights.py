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

SYSTEM_INSTRUCTION = (
    "You are a genomics analyst. Analyze the given genomic KPIs and "
    "sequence motifs and provide a plain-language summary of the "
    "biological significance, focusing on potential pathogenic "
    "anomalies or regulatory features. Be concise and avoid overclaiming "
    "clinical significance from summary statistics alone."
)


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
    
    max_attempts = 5
    base_wait = 2  # Start with 2 second wait
    
    for attempt in range(max_attempts):
        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                if attempt < max_attempts - 1:
                    # Check for Retry-After header
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after:
                        wait_time = int(retry_after)
                    else:
                        # Exponential backoff: 2, 4, 8, 16 seconds
                        wait_time = base_wait * (2 ** attempt)
                    
                    print(f"Rate limited (429). Waiting {wait_time}s before retry {attempt + 1}/{max_attempts}...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(
                        f"API rate limit exceeded after {max_attempts} retries. "
                        "Please wait a few minutes and try again."
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
    
    max_attempts = 5
    base_wait = 2  # Start with 2 second wait
    
    for attempt in range(max_attempts):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return "".join(block["text"] for block in data["content"] if block["type"] == "text")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                if attempt < max_attempts - 1:
                    # Check for Retry-After header
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after:
                        wait_time = int(retry_after)
                    else:
                        # Exponential backoff: 2, 4, 8, 16 seconds
                        wait_time = base_wait * (2 ** attempt)
                    
                    print(f"Rate limited (429). Waiting {wait_time}s before retry {attempt + 1}/{max_attempts}...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(
                        f"API rate limit exceeded after {max_attempts} retries. "
                        "Please wait a few minutes and try again."
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
