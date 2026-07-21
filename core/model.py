"""
Genomic Foundation Model (GFM) integration.

Loads a Hugging Face model such as DNABERT-2 (zhihan1996/DNABERT-2-117M)
or HyenaDNA, cached as a global singleton via st.cache_resource so the
(large) weights are only pulled into memory once per server process —
not once per user session or per Streamlit rerun.

NOTE: downloading model weights requires outbound internet access to
huggingface.co. If you're running behind a restricted network, set
HF_HUB_OFFLINE=1 and pre-download the weights, or point
TRANSFORMERS_CACHE at a volume where they're already cached.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np

try:
    import streamlit as st
    _cache_resource = st.cache_resource
except ImportError:  # allows this module to be imported/tested outside Streamlit
    def _cache_resource(func):
        return lru_cache(maxsize=1)(func)


DEFAULT_MODEL_NAME = "zhihan1996/DNABERT-2-117M"


@_cache_resource
def load_gfm(model_name: str = DEFAULT_MODEL_NAME):
    """
    Loads tokenizer + model once and caches them as a singleton for the
    life of the server process. Subsequent calls (across all user
    sessions) return the cached object instead of re-loading weights.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    model.eval()

    return {"tokenizer": tokenizer, "model": model, "device": device}


def embed_sequences(sequences: List[str], model_bundle: dict, batch_size: int = 8) -> np.ndarray:
    """
    Extract fixed-length embeddings for a list of DNA sequences using
    mean token pooling over the model's last hidden state. Returns an
    (n_sequences, hidden_dim) numpy array suitable for downstream
    classification / clustering.
    """
    import torch

    tokenizer = model_bundle["tokenizer"]
    model = model_bundle["model"]
    device = model_bundle["device"]

    all_embeddings = []

    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start:start + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            outputs = model(**inputs)
            hidden_states = outputs[0]  # (batch, seq_len, hidden_dim)

            # Mean pooling, respecting the attention mask so padding
            # tokens don't dilute the embedding.
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            summed = (hidden_states * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = (summed / counts).cpu().numpy()

            all_embeddings.append(pooled)

    return np.vstack(all_embeddings) if all_embeddings else np.empty((0, 0))


def gpu_tokenize_hint() -> str:
    """
    For single-nucleotide-resolution workloads (e.g. per-base variant
    scoring across long sequences), plain CPU tokenization becomes the
    bottleneck. This returns guidance text shown in the UI rather than
    silently switching tokenizers, since it requires an extra dependency
    (dnatok) the user must opt into.
    """
    return (
        "For single-nucleotide-resolution analysis, CPU tokenization is "
        "typically the bottleneck. Consider installing a GPU-accelerated "
        "tokenizer (e.g. `dnatok`) and routing `tokenizer.__call__` "
        "through it to keep the GPU fed during large batch runs."
    )


# ---------------------------------------------------------------------------
# Hosted-API embeddings — no local model download required.
#
# IMPORTANT: DNABERT-2 (and most DNA-specific foundation models, e.g.
# InstaDeepAI's nucleotide-transformer family) are NOT deployed on Hugging
# Face's hosted Inference Providers as of writing — they use custom
# modeling code (`trust_remote_code=True`), which HF's serverless API does
# not execute for security reasons. So a genuinely "no download" option
# means calling a *general-purpose* embedding model that IS hosted (e.g.
# sentence-transformers/all-MiniLM-L6-v2), treating the DNA string as
# plain text. It won't carry DNABERT-2's genomic pretraining, but it's a
# real, callable, zero-download embedding.
#
# If a DNA-specific model becomes available on Inference Providers later,
# just point HOSTED_MODEL_NAME at it — the call pattern is the same.
# ---------------------------------------------------------------------------

HOSTED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HF_ROUTER_URL = "https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"


def embed_sequences_openai_api(
    sequences: List[str],
    api_key: str,
    model_name: str = OPENAI_EMBEDDING_MODEL,
    batch_size: int = 16,
    timeout: int = 30,
) -> np.ndarray:
    """
    Get text embeddings from OpenAI's embedding API.
    Returns an (n_sequences, hidden_dim) numpy array suitable for downstream
    analysis or clustering.
    """
    import requests

    if not api_key:
        raise RuntimeError(
            "An OpenAI API key is required for hosted embeddings. "
            "Set OPENAI_API_KEY in .streamlit/secrets.toml or as an environment variable."
        )

    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    all_embeddings = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start + batch_size]
        resp = requests.post(
            url,
            headers=headers,
            json={"input": batch, "model": model_name},
            timeout=timeout,
        )
        if resp.status_code == 401:
            raise RuntimeError("OpenAI API key was rejected (401) — check the token value.")
        resp.raise_for_status()

        payload = resp.json()
        batch_embeddings = [item["embedding"] for item in payload.get("data", [])]
        all_embeddings.extend(batch_embeddings)

    return np.asarray(all_embeddings, dtype=np.float32) if all_embeddings else np.empty((0, 0))


def embed_sequences_hosted_api(
    sequences: List[str],
    api_key: str,
    model_name: str = HOSTED_MODEL_NAME,
    batch_size: int = 16,
    timeout: int = 30,
) -> np.ndarray:
    """
    Get sequence embeddings from Hugging Face's hosted Inference API —
    no model weights are ever downloaded to this machine. Requires a
    (free) HF access token: https://huggingface.co/settings/tokens

    Returns an (n_sequences, hidden_dim) numpy array, same shape/contract
    as the local embed_sequences() function above, so callers can swap
    between the two without changing downstream code.
    """
    import time

    import requests

    if not api_key:
        raise RuntimeError(
            "A Hugging Face API token is required for hosted embeddings. "
            "Get a free one at https://huggingface.co/settings/tokens"
        )

    url = HF_ROUTER_URL.format(model=model_name)
    headers = {"Authorization": f"Bearer {api_key}"}

    all_embeddings = []

    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start + batch_size]

        # Hosted models loaded on-demand can return 503 with an
        # estimated_time while they spin up; retry a few times.
        for attempt in range(4):
            resp = requests.post(url, headers=headers, json={"inputs": batch}, timeout=timeout)
            if resp.status_code == 503:
                wait_s = min(resp.json().get("estimated_time", 5), 20)
                time.sleep(wait_s)
                continue
            break

        if resp.status_code == 401:
            raise RuntimeError("Hugging Face API token was rejected (401) — check the token value.")
        if resp.status_code == 404:
            raise RuntimeError(
                f"Model '{model_name}' isn't available on HF's hosted Inference API "
                "(404). Pick a different, hosted feature-extraction model."
            )
        resp.raise_for_status()

        batch_vectors = resp.json()
        # Some models return per-token vectors (list of lists per input);
        # mean-pool across tokens to match the local pipeline's output shape.
        pooled_batch = []
        for vec in batch_vectors:
            arr = np.array(vec, dtype=np.float32)
            if arr.ndim == 2:  # (tokens, hidden_dim) -> mean pool
                arr = arr.mean(axis=0)
            pooled_batch.append(arr)
        all_embeddings.extend(pooled_batch)

    return np.vstack(all_embeddings) if all_embeddings else np.empty((0, 0))
