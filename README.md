# Genomic Dashboard

A high-performance, extensible dashboard for exploring gigabyte-scale
genomic data, with optional foundation-model (DNABERT-2 / HyenaDNA)
embeddings and LLM-generated plain-language insights.

## Features

- **In-memory streaming FASTA/FASTQ parser** — handles multi-GB files
  without writing to disk, constant-ish memory footprint via generators.
- **Live KPIs** — record count, total bases, GC content, length
  distribution, ambiguous-base rate — computed with running counters,
  not full in-memory storage.
- **Interactive sequence viewer** — linear/circular DNA view (via
  seqviz.js).
- **3D structure viewer** — cartoon rendering of uploaded PDB files
  (via NGL.js), for AlphaFold/ESMFold output.
- **Genomic Foundation Model embeddings** — optional DNABERT-2
  integration via Hugging Face `transformers`, cached as a singleton
  so weights load once per server process.
- **AI Insights** — sends KPI summaries to Gemini or Claude for a
  plain-language biological interpretation.
- **FastAPI backend** — decouples heavy inference from the UI process,
  with a minimal role-based access control example and audit logging
  scaffold (a starting point toward 21 CFR Part 11-style logging, not
  a compliance guarantee).

## Project layout

```
genomic_dashboard/
├── app.py                  # Streamlit front end (run this)
├── requirements.txt
├── .streamlit/config.toml  # theme + 4GB max upload size
├── core/
│   ├── parser.py            # streaming FASTA/FASTQ parsing
│   ├── kpis.py               # running KPI aggregation
│   ├── model.py               # GFM loading + embedding extraction
│   └── ai_insights.py          # Gemini/Claude KPI summarization
├── components/
│   ├── seqviz.py             # linear/circular sequence viewer (CDN JS)
│   └── ngl_viewer.py          # 3D structure viewer (CDN JS)
└── backend/
    └── main.py               # FastAPI service (optional, separate process)
```

## Quickstart

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
```

Open the local URL Streamlit prints (defaults to `http://localhost:8501`).

Upload any `.fasta`/`.fastq` file in the sidebar — KPIs, distribution
charts, and the sequence viewer populate automatically.

### Enabling sequence embeddings

Two modes, selectable in the sidebar:

**Hosted API (no download)** — calls Hugging Face's Inference API over
HTTPS. Nothing is installed or downloaded to your machine. You need a
free HF token (https://huggingface.co/settings/tokens), pasted into
the sidebar field. Note the trade-off: DNA-specific foundation models
like DNABERT-2 aren't available on HF's hosted Inference Providers (as
of writing they require custom modeling code, which the serverless API
won't execute) — so this mode uses a general-purpose embedding model
(`sentence-transformers/all-MiniLM-L6-v2` by default) applied to the
raw sequence text. It's genuinely zero-download, but not
genomics-pretrained. If a DNA-specific model becomes hosted later,
just change `HOSTED_MODEL_NAME` in `core/model.py`.

**Local model (downloads weights)** — downloads DNABERT-2 (~117M
params) from Hugging Face Hub the first time it runs, then caches it
as a singleton via `st.cache_resource`. Genuinely genomic-aware
embeddings, but needs internet access for that first download and a
GPU is recommended for anything beyond small batches.

```bash
export HF_TOKEN=your-token   # optional: pre-set instead of pasting in the sidebar
```

### Enabling AI Insights

Set one of these environment variables before launching:

```bash
export GEMINI_API_KEY=your-key      # for the Gemini provider
export ANTHROPIC_API_KEY=your-key   # for the Claude provider
```

Pick the provider in the sidebar, then click "Generate plain-language
summary" under AI Insights.

### Running the FastAPI backend separately (optional)

For multi-user deployments, run inference as its own service so the
Streamlit process(es) stay lightweight:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Demo API keys are hardcoded for illustration only
(`demo-analyst-key`, `demo-admin-key` in `backend/main.py`) — replace
`API_KEY_ROLES` with a real identity provider (OIDC/JWT, your
institution's SSO, etc.) before handling real patient data. Same goes
for the audit logger, which currently writes to stdout/log file; swap
in an append-only store for anything regulatory.

## Notes on the original spec

A couple of adjustments from the initial prompt list, made during
implementation:

- **`st-seqviz`** isn't a real, maintained PyPI package as far as I
  could verify, so the sequence viewer embeds the `seqviz` JavaScript
  library directly via CDN inside a Streamlit HTML component instead.
  Same approach for the NGL 3D viewer.
- **`DNAtok`** (GPU-accelerated tokenizer) is mentioned in the UI as a
  recommendation rather than wired in automatically, since it's an
  extra dependency you'd need to opt into deliberately.
- The FastAPI RBAC/audit-log code is a *pattern*, not a compliance
  certification — treat the 21 CFR Part 11 mention in the original
  spec as "build toward this," not "this satisfies it out of the box."

## Scaling tips

- Use `st.cache_data` for DataFrame-shaped results, `st.cache_resource`
  for shared objects like loaded models (both already used here).
- On Hugging Face Spaces or similar ephemeral-disk platforms, set
  `HF_HOME` / `TRANSFORMERS_CACHE` to a persistent volume so model
  weights don't re-download on every restart.
- For simple classification tasks, embedding + a lightweight
  classifier (e.g. logistic regression / small MLP on top of pooled
  embeddings) is typically 10-20x faster than fine-tuning the full
  foundation model, and cheaper to serve.
