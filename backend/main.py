"""
FastAPI backend that decouples heavy model inference from the Streamlit
front end, so multiple concurrent dashboard users share one inference
service instead of each loading model weights into their own process.

Run with:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000

The Streamlit app talks to this over HTTP (see core/model.py for the
in-process fallback used when you don't want a separate service).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("genomic_backend")

app = FastAPI(title="Genomic Dashboard Backend", version="1.0.0")

# ---------------------------------------------------------------------------
# Role-based access control
#
# In production, replace this static map with a real identity provider
# (OIDC/JWT via e.g. Auth0, Okta, or your institution's SSO). This is a
# minimal illustrative version: API keys mapped to roles.
# ---------------------------------------------------------------------------
API_KEY_ROLES = {
    "demo-analyst-key": "analyst",
    "demo-admin-key": "admin",
}

ROLE_PERMISSIONS = {
    "analyst": {"embed", "read_kpis"},
    "admin": {"embed", "read_kpis", "manage_models"},
}


def get_current_role(x_api_key: Optional[str] = Header(default=None)) -> str:
    if x_api_key is None or x_api_key not in API_KEY_ROLES:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return API_KEY_ROLES[x_api_key]


def require_permission(permission: str):
    def _checker(role: str = Depends(get_current_role)) -> str:
        if permission not in ROLE_PERMISSIONS.get(role, set()):
            raise HTTPException(status_code=403, detail=f"Role '{role}' lacks '{permission}' permission")
        return role
    return _checker


# ---------------------------------------------------------------------------
# Audit logging — a minimal stand-in for FDA 21 CFR Part 11-style
# electronic-record logging. In a real deployment this should write to an
# append-only, tamper-evident store (e.g. a WORM S3 bucket or a signed log
# table), not a local log file.
# ---------------------------------------------------------------------------
def audit_log(action: str, role: str, detail: str = "") -> None:
    logger.info(
        "AUDIT ts=%s action=%s role=%s detail=%s",
        datetime.utcnow().isoformat(),
        action,
        role,
        detail,
    )


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class EmbedRequest(BaseModel):
    sequences: List[str]
    model_name: Optional[str] = "zhihan1996/DNABERT-2-117M"


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dim: int
    latency_ms: float


class KPIRequest(BaseModel):
    sequences: List[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest, role: str = Depends(require_permission("embed"))):
    from core.model import embed_sequences, load_gfm

    start = time.perf_counter()
    bundle = load_gfm(req.model_name)
    vectors = embed_sequences(req.sequences, bundle)
    latency_ms = (time.perf_counter() - start) * 1000

    audit_log("embed", role, detail=f"n_sequences={len(req.sequences)} model={req.model_name}")

    return EmbedResponse(
        embeddings=vectors.tolist(),
        dim=vectors.shape[1] if vectors.size else 0,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/kpis")
def kpis(req: KPIRequest, role: str = Depends(require_permission("read_kpis"))):
    from core.kpis import RunningKPIs
    from core.parser import SequenceRecord

    running = RunningKPIs()
    for i, seq in enumerate(req.sequences):
        running.update(SequenceRecord(id=str(i), sequence=seq))

    audit_log("read_kpis", role, detail=f"n_sequences={len(req.sequences)}")
    return running.summary_dict()
