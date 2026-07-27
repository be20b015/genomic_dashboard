"""
Genomic Dashboard — main Streamlit entry point.

Run with:
    streamlit run app.py
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Allow `core` / `components` package imports when run as `streamlit run app.py`
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import plotly.express as px
import streamlit as st


def _get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        if hasattr(st, "secrets") and st.secrets:
            for key in (name, name.upper(), name.lower(), name.replace("-", "_")):
                try:
                    value = st.secrets[key]
                    if value:
                        return str(value)
                except Exception:
                    continue
    except Exception:
        pass

    for key in (name, name.upper(), name.lower(), name.replace("-", "_")):
        value = os.getenv(key)
        if value:
            return value
    return default


def _load_runtime_secrets() -> None:
    preferred_values = {
        "AWS_ACCESS_KEY_ID": _get_secret("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": _get_secret("AWS_SECRET_ACCESS_KEY"),
        "AWS_SESSION_TOKEN": _get_secret("AWS_SESSION_TOKEN"),
        "AWS_REGION": _get_secret("AWS_REGION", "us-east-1"),
        "BEDROCK_CLAUDE_MODEL_ID": _get_secret(
            "BEDROCK_CLAUDE_MODEL_ID",
            "anthropic.claude-sonnet-4-6-v1:0",
        ),
    }

    for key, value in preferred_values.items():
        if value:
            os.environ[key] = str(value)


from components.ngl_viewer import render_structure
from components.seqviz import render_seqviz
from core.kpis import RunningKPIs
from core.parser import stream_with_progress

st.set_page_config(page_title="Genomic Dashboard", layout="wide", page_icon="🧬")
_load_runtime_secrets()

# ---------------------------------------------------------------------------
# Sidebar — file upload & options
# ---------------------------------------------------------------------------
st.sidebar.title("🧬 Genomic Dashboard")
uploaded_file = st.sidebar.file_uploader(
    "Upload FASTA or FASTQ file",
    type=["fasta", "fa", "fastq", "fq", "txt"],
    help="Files are parsed in-memory (streamed) — nothing is written to disk.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("AI Model")
enable_ai = st.sidebar.checkbox(
    "Enable sequence embeddings",
    value=False,
    help="Turn a sequence into a numeric vector for downstream classification/clustering.",
)
embedding_mode = st.sidebar.radio(
    "Embedding source",
    ["Hosted API (no download)", "Local model (downloads weights)"],
    index=0,
    help=(
        "Hosted API: calls a hosted embedding provider, nothing installed "
        "or downloaded locally — needs an API key. Note: DNA-specific "
        "models like DNABERT-2 aren't hosted on every provider, so this "
        "uses a general-purpose text embedding model on the raw sequence text.\n\n"
        "Local model: downloads DNABERT-2 weights (~117M params) and runs "
        "them on this machine — genuinely genomic-aware embeddings, but "
        "needs internet access for the first download and a GPU is "
        "recommended for large batches."
    ),
) if enable_ai else None
hosted_embedding_provider = None
hosted_api_key = None
if enable_ai and embedding_mode == "Hosted API (no download)":
    hosted_embedding_provider = st.sidebar.selectbox(
        "Hosted embedding provider",
        ["Hugging Face", "OpenAI"],
        index=0,
        help="Choose which hosted provider should generate the embeddings.",
    )

    if hosted_embedding_provider == "OpenAI":
        hosted_api_key = _get_secret("OPENAI_API_KEY")
        hosted_api_key = st.sidebar.text_input(
            "OpenAI API key",
            value=hosted_api_key or "",
            type="password",
            help="Used for hosted text embeddings. Stored securely via Streamlit secrets or environment variables.",
        ) or hosted_api_key
    else:
        hosted_api_key = None
        for secret_name in ("HF_API_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_API_TOKEN"):
            hosted_api_key = _get_secret(secret_name)
            if hosted_api_key:
                break
        hosted_api_key = st.sidebar.text_input(
            "Hugging Face API token",
            value=hosted_api_key or "",
            type="password",
            help="Free token from https://huggingface.co/settings/tokens. Stored securely via Streamlit secrets or environment variables.",
        ) or hosted_api_key

st.sidebar.markdown("---")
uploaded_pdb = st.sidebar.file_uploader(
    "Optional: 3D structure (.pdb)",
    type=["pdb"],
    help="e.g. AlphaFold / ESMFold prediction output",
)

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Genomic Data Dashboard")

if uploaded_file is None:
    st.info("Upload a FASTA or FASTQ file in the sidebar to get started.")
    st.stop()

raw_bytes = uploaded_file.getvalue()  # buffered in RAM by Streamlit, no disk write
size_mb = len(raw_bytes) / (1024 * 1024)
st.caption(f"Loaded **{uploaded_file.name}** ({size_mb:.1f} MB) — streaming parse in progress…")

progress_bar = st.progress(0.0)
status_text = st.empty()

running = RunningKPIs()
first_record = None
records_for_viewer = []

def _on_progress(n_records, pct):
    progress_bar.progress(pct)
    status_text.text(f"Parsed {n_records:,} records…")

try:
    for record in stream_with_progress(raw_bytes, progress_callback=_on_progress):
        running.update(record)
        if first_record is None:
            first_record = record
        if len(records_for_viewer) < 200:  # cap what we keep for the interactive viewer
            records_for_viewer.append(record)
except ValueError as e:
    st.error(str(e))
    st.stop()

progress_bar.empty()
status_text.empty()
st.success(f"Parsed {running.n_records:,} records ({running.total_length:,} total bases).")

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
summary = running.summary_dict()
kpi_cols = st.columns(6)
kpi_cols[0].metric("Records", f"{summary['records']:,}")
kpi_cols[1].metric("Total bases", f"{summary['total_bases']:,}")
kpi_cols[2].metric("Mean length", summary["mean_length"])
kpi_cols[3].metric("GC content", f"{summary['overall_gc_pct']}%")
kpi_cols[4].metric("Min / Max length", f"{summary['min_length']} / {summary['max_length']}")
kpi_cols[5].metric("Ambiguous bases", f"{summary['ambiguous_base_pct']}%")

# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------
st.subheader("Distributions")
dist_col1, dist_col2 = st.columns(2)

with dist_col1:
    len_df = pd.DataFrame({"length": running.length_samples})
    fig = px.histogram(len_df, x="length", nbins=40, title="Sequence length distribution")
    st.plotly_chart(fig, use_container_width=True)

with dist_col2:
    gc_df = pd.DataFrame({"gc_content": running.gc_samples})
    fig2 = px.histogram(gc_df, x="gc_content", nbins=40, title="Per-record GC content (%)")
    st.plotly_chart(fig2, use_container_width=True)

if len(running.length_samples) < running.n_records:
    st.caption(
        f"Distribution plots are based on a {len(running.length_samples):,}-record "
        f"sample (of {running.n_records:,} total) to keep memory bounded."
    )

# ---------------------------------------------------------------------------
# Sequence viewer
# ---------------------------------------------------------------------------
st.subheader("Sequence Viewer")
if records_for_viewer:
    viewer_ids = [r.id for r in records_for_viewer]
    selected_id = st.selectbox("Select a record to view", viewer_ids)
    selected_record = next(r for r in records_for_viewer if r.id == selected_id)
    layout = st.radio("Layout", ["linear", "circular"], horizontal=True)
    render_seqviz(selected_record.sequence, name=selected_record.id, viewer_type=layout)

# ---------------------------------------------------------------------------
# 3D structure viewer
# ---------------------------------------------------------------------------
if uploaded_pdb is not None:
    st.subheader("3D Protein Structure")
    pdb_text = uploaded_pdb.getvalue().decode("utf-8", errors="ignore")
    render_structure(pdb_text)

# ---------------------------------------------------------------------------
# Sequence embeddings (optional) — hosted API or local model download
# ---------------------------------------------------------------------------
if enable_ai:
    st.subheader("Sequence Embeddings")
    if embedding_mode == "Hosted API (no download)":
        st.caption(
            "Using Hugging Face's hosted Inference API — no model is downloaded "
            "to this machine. Note: this uses a general-purpose text embedding "
            "model, not a DNA-specific one (see sidebar tooltip for why)."
        )
    else:
        st.caption("Using a locally downloaded DNABERT-2 model.")

    available = len(records_for_viewer)
    if available <= 1:
        # st.slider requires min_value < max_value; with 0 or 1 records
        # available there's no real range to pick from, so skip the widget.
        n_to_embed = available
        st.caption(f"{available} sequence available — will embed all of it." if available else "No sequences available to embed.")
    else:
        max_selectable = min(50, available)
        default_value = min(8, max_selectable)
        n_to_embed = st.slider("Number of sequences to embed", 1, max_selectable, default_value)

    run_disabled = n_to_embed < 1 or (embedding_mode == "Hosted API (no download)" and not hosted_api_key)
    if run_disabled and n_to_embed >= 1 and embedding_mode == "Hosted API (no download)":
        provider_name = hosted_embedding_provider or "hosted provider"
        st.caption(f"Enter an API key for {provider_name} in the sidebar to enable this.")

    if st.button("Run embedding extraction", disabled=run_disabled):
        seqs = [r.sequence for r in records_for_viewer[:n_to_embed]]

        if embedding_mode == "Hosted API (no download)":
            with st.spinner("Calling hosted embedding API…"):
                try:
                    if hosted_embedding_provider == "OpenAI":
                        from core.model import embed_sequences_openai_api

                        vectors = embed_sequences_openai_api(seqs, api_key=hosted_api_key)
                    else:
                        from core.model import embed_sequences_hosted_api

                        vectors = embed_sequences_hosted_api(seqs, api_key=hosted_api_key)

                    st.write(f"Embedding matrix shape: {vectors.shape}")
                    st.dataframe(pd.DataFrame(vectors[:, :10]).round(4), use_container_width=True)
                except Exception as e:
                    st.error(f"Hosted embedding request failed: {e}")
        else:
            with st.spinner("Loading model and extracting embeddings (first load may take a while)…"):
                try:
                    from core.model import embed_sequences, load_gfm, gpu_tokenize_hint

                    bundle = load_gfm()
                    vectors = embed_sequences(seqs, bundle)
                    st.write(f"Embedding matrix shape: {vectors.shape}")
                    st.dataframe(pd.DataFrame(vectors[:, :10]).round(4), use_container_width=True)
                    st.caption(gpu_tokenize_hint())
                except Exception as e:
                    st.error(
                        "Could not load the foundation model. This usually means "
                        "there's no internet access to huggingface.co in this "
                        f"environment, or `transformers`/`torch` aren't installed.\n\nDetails: {e}"
                    )

# ---------------------------------------------------------------------------
# AI Insights (Amazon Bedrock Claude)
# ---------------------------------------------------------------------------
st.subheader("AI Insights")
st.caption("Uses Amazon Bedrock Claude with credentials loaded from Streamlit secrets or your local environment.")
ai_col1, ai_col2 = st.columns(2)

with ai_col1:
    if st.button("Generate KPI Summary", key="kpi_summary"):
        with st.spinner("Asking Amazon Bedrock Claude for an interpretation…"):
            try:
                from core.ai_insights import get_insights

                text = get_insights(summary, provider="claude")
                st.markdown(text)
            except RuntimeError as e:
                st.warning(
                    f"{e}. Configure AWS credentials and region before running the app."
                )
            except Exception as e:
                st.error(f"AI insights request failed: {e}")

with ai_col2:
    if st.button("Analyze Individual Sequence", key="fasta_analysis"):
        if records_for_viewer:
            viewer_ids = [r.id for r in records_for_viewer]
            analysis_id = st.selectbox("Select sequence to analyze", viewer_ids, key="analyze_select")
            analysis_record = next(r for r in records_for_viewer if r.id == analysis_id)

            with st.spinner(f"Analyzing sequence {analysis_id} with Amazon Bedrock Claude…"):
                try:
                    from core.ai_insights import analyze_fasta_sequence

                    result = analyze_fasta_sequence(analysis_id, analysis_record.sequence)

                    st.json(result)

                    with st.expander("📋 Sequence Analysis Summary", expanded=True):
                        if "sequence_metadata" in result:
                            st.write("**Sequence Metadata:**")
                            st.write(result["sequence_metadata"])

                        if "nucleotide_composition" in result:
                            st.write("**Nucleotide Composition:**")
                            st.write(result["nucleotide_composition"])

                        if "open_reading_frames" in result:
                            st.write("**Open Reading Frames:**")
                            for orf in result["open_reading_frames"]:
                                st.write(f"- {orf}")

                        if "sequence_features" in result:
                            st.write("**Sequence Features:**")
                            for feature in result["sequence_features"]:
                                st.write(f"- {feature}")

                        if "clinical_or_biological_relevance" in result:
                            st.write("**Clinical/Biological Relevance:**")
                            st.write(result["clinical_or_biological_relevance"])

                        if "quality_warnings" in result and result["quality_warnings"]:
                            st.warning("**Quality Warnings:**")
                            for warning in result["quality_warnings"]:
                                st.write(f"⚠️ {warning}")

                except RuntimeError as e:
                    st.warning(
                        f"{e}. Configure AWS credentials and region before running the app."
                    )
                except Exception as e:
                    st.error(f"FASTA sequence analysis failed: {e}")
        else:
            st.info("No sequences available to analyze. Upload a file first.")
