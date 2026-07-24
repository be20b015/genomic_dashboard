"""
Clean, provider-agnostic AI-insight module for parsed-genome KPIs and
FASTA sequence analysis.

This trims the original module down to what's actually needed and adds
a Claude-based structured-JSON analyzer (mirroring the Gemini one) plus
a markdown formatter that renders the same clean, numbered-section
layout you get from a good structured response.

Claude is called via Amazon Bedrock (boto3), not the direct Anthropic
API — so auth is standard AWS credentials (env vars, ~/.aws/credentials,
an IAM role, etc.), not an ANTHROPIC_API_KEY.

Requires: boto3  (pip install boto3)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Optional

import boto3
from botocore.exceptions import ClientError

# Bedrock model ID for Claude. Swap for the inference-profile ARN/ID your
# account uses if on-demand throughput isn't enabled for this model in
# your region (e.g. "us.anthropic.claude-sonnet-4-6-v1:0").
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_CLAUDE_MODEL_ID", "anthropic.claude-sonnet-4-6-v1:0")
BEDROCK_REGION = os.environ.get("AWS_REGION", "us-east-1")

_bedrock_client = None


def _get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    return _bedrock_client


SYSTEM_INSTRUCTION = (
    "You are a genomics analyst. Analyze the given genomic KPIs and "
    "sequence motifs and provide a plain-language summary of the "
    "biological significance, focusing on potential pathogenic "
    "anomalies or regulatory features. Be concise and avoid overclaiming "
    "clinical significance from summary statistics alone."
)

MIN_REQUEST_INTERVAL = 2  # seconds between requests per provider

_insights_cache: dict[str, tuple[float, str]] = {}
_last_request_time: dict[str, float] = {}


# --------------------------------------------------------------------------
# Small shared helpers
# --------------------------------------------------------------------------

def _cache_key(*parts: str) -> str:
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def _get_cached(key: str, max_age: int = 3600) -> Optional[str]:
    hit = _insights_cache.get(key)
    if not hit:
        return None
    ts, value = hit
    if time.time() - ts >= max_age:
        del _insights_cache[key]
        return None
    return value


def _set_cached(key: str, value: str) -> None:
    _insights_cache[key] = (time.time(), value)


def _throttle(provider: str) -> None:
    last = _last_request_time.get(provider)
    if last is not None:
        wait = MIN_REQUEST_INTERVAL - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
    _last_request_time[provider] = time.time()


def _invoke_claude_bedrock(system: Optional[str], user_content: str, max_tokens: int,
                            temperature: float = 1.0, max_attempts: int = 3,
                            base_wait: int = 5) -> str:
    """
    Call Claude on Bedrock via invoke_model, with exponential-backoff retry
    on throttling. Returns the concatenated text of the response.
    """
    client = _get_bedrock_client()

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": user_content}],
    }
    if system:
        body["system"] = system

    for attempt in range(max_attempts):
        try:
            resp = client.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            payload = json.loads(resp["body"].read())
            return "".join(b["text"] for b in payload.get("content", []) if b.get("type") == "text")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("ThrottlingException", "TooManyRequestsException") and attempt < max_attempts - 1:
                wait = base_wait * (2 ** attempt)
                time.sleep(wait)
                continue
            raise RuntimeError(f"Bedrock error ({error_code}): {e.response.get('Error', {}).get('Message', str(e))}")
    raise RuntimeError(f"Failed to get a response from Bedrock after {max_attempts} attempts")


# --------------------------------------------------------------------------
# Plain-language KPI insight (unchanged behavior, trimmed)
# --------------------------------------------------------------------------

def _build_prompt(kpi_summary: dict, extra_context: str = "") -> str:
    lines = [f"{k}: {v}" for k, v in kpi_summary.items()]
    prompt = "Genomic KPIs:\n" + "\n".join(lines)
    if extra_context:
        prompt += f"\n\nAdditional context:\n{extra_context}"
    return prompt


def get_insights_claude(kpi_summary: dict, extra_context: str = "") -> str:
    key = _cache_key("claude", str(sorted(kpi_summary.items())), extra_context)
    cached = _get_cached(key)
    if cached:
        return cached

    _throttle("claude")

    result = _invoke_claude_bedrock(
        system=SYSTEM_INSTRUCTION,
        user_content=_build_prompt(kpi_summary, extra_context),
        max_tokens=500,
    )
    _set_cached(key, result)
    return result


# --------------------------------------------------------------------------
# Structured FASTA analysis via Claude (mirrors the Gemini structured-JSON
# version, but returns the exact same schema so downstream code doesn't
# need to branch on provider)
# --------------------------------------------------------------------------

_FASTA_SCHEMA_PROMPT = """\
You are a bioinformatics annotation assistant. Analyze the following FASTA \
sequence and respond with ONLY a JSON object (no markdown fences, no \
preamble) matching exactly this schema:

{{
  "sequence_metadata": {{
    "organism_guess": string,
    "genome_type": string,
    "sequence_length_bp": integer
  }},
  "nucleotide_composition": {{
    "estimated_gc_content_pct": number,
    "degenerate_bases_found": [string]
  }},
  "open_reading_frames": [
    {{"name": string, "region": string}}
  ],
  "sequence_features": [string],
  "clinical_or_biological_relevance": string,
  "quality_warnings": [string]
}}

Base every field only on what is inferable from the sequence and header \
below. If something can't be determined, use an empty string, empty \
array, or 0 as appropriate — do not fabricate specifics.

Header: {header}
Sequence: {sequence}
"""


def analyze_fasta_sequence_claude(fasta_header: str, dna_sequence: str) -> dict:
    """
    Structured JSON annotation of a FASTA sequence via Claude on Bedrock,
    matching the same schema as the Gemini-based analyze_fasta_sequence().
    """
    key = _cache_key("fasta-claude", fasta_header, dna_sequence[:500])
    cached = _get_cached(key, max_age=7200)
    if cached:
        return json.loads(cached)

    _throttle("fasta-analysis-claude")

    sequence_display = dna_sequence[:3000]
    if len(dna_sequence) > 3000:
        sequence_display += f"... [truncated, original length: {len(dna_sequence)} bp]"

    prompt = _FASTA_SCHEMA_PROMPT.format(header=fasta_header, sequence=sequence_display)

    raw_text = _invoke_claude_bedrock(
        system=None,
        user_content=prompt,
        max_tokens=1200,
        temperature=0.1,
    ).strip()

    # Strip stray ```json fences if the model adds them anyway
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.lower().startswith("json"):
            raw_text = raw_text[4:].strip()

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse Claude response as JSON: {e}\nRaw: {raw_text[:500]}")

    _set_cached(key, json.dumps(result))
    return result


# --------------------------------------------------------------------------
# Formatter: turns the structured dict into the clean numbered-section
# markdown layout (organism/classification -> ORFs -> features)
# --------------------------------------------------------------------------

def format_fasta_analysis_markdown(result: dict, sequence_label: str = "sequence") -> str:
    """Render analyze_fasta_sequence()-style output as clean markdown."""
    meta = result.get("sequence_metadata", {})
    comp = result.get("nucleotide_composition", {})
    orfs = result.get("open_reading_frames", [])
    features = result.get("sequence_features", [])
    relevance = result.get("clinical_or_biological_relevance", "")
    warnings = result.get("quality_warnings", [])

    lines = [f"Based on the sequence provided, here is the analysis of `{sequence_label}`:", ""]

    lines.append("**1. Sequence Identification & Classification**")
    lines.append("")
    lines.append(f"* Organism / Source: {meta.get('organism_guess', 'Unknown')}.")
    lines.append(f"* Genome Type: {meta.get('genome_type', 'Unknown')}.")
    lines.append(f"* Length: approximately {meta.get('sequence_length_bp', 'N/A')} bp.")
    lines.append("")

    lines.append("**2. Genomic Organization & Key Open Reading Frames (ORFs)**")
    lines.append("")
    if orfs:
        for orf in orfs:
            name = orf.get("name", "Unnamed ORF")
            region = orf.get("region", "region not specified")
            lines.append(f"* `{name}`: {region}.")
    else:
        lines.append("* No ORFs reported.")
    lines.append("")

    lines.append("**3. Notable Sequence Features & Motifs**")
    lines.append("")
    if features:
        for f in features:
            lines.append(f"* {f}")
    else:
        lines.append("* None reported.")
    lines.append("")

    gc = comp.get("estimated_gc_content_pct")
    degenerate = comp.get("degenerate_bases_found", [])
    lines.append("**4. Nucleotide Composition**")
    lines.append("")
    lines.append(f"* Estimated GC content: {gc if gc is not None else 'N/A'}%.")
    lines.append(f"* Degenerate/ambiguous bases found: {', '.join(degenerate) if degenerate else 'None'}.")
    lines.append("")

    if relevance:
        lines.append("**5. Biological / Research Relevance**")
        lines.append("")
        lines.append(relevance)
        lines.append("")

    if warnings:
        lines.append("**6. Quality Warnings**")
        lines.append("")
        for w in warnings:
            lines.append(f"* {w}")
        lines.append("")

    return "\n".join(lines).strip()


# --------------------------------------------------------------------------
# Unified entry points
# --------------------------------------------------------------------------

def get_insights(kpi_summary: dict, extra_context: str = "", provider: str = "claude") -> str:
    if provider == "claude":
        return get_insights_claude(kpi_summary, extra_context)
    raise ValueError(f"Unknown provider: {provider}")


def analyze_fasta_sequence(fasta_header: str, dna_sequence: str, as_markdown: bool = False):
    """
    Structured FASTA analysis via Claude on Bedrock. Returns a dict by
    default, or a clean markdown string if as_markdown=True.
    """
    result = analyze_fasta_sequence_claude(fasta_header, dna_sequence)
    if as_markdown:
        return format_fasta_analysis_markdown(result, sequence_label=fasta_header)
    return result