"""
Schemas untuk endpoint /v1/health dan /v1/health/detailed.

  HealthResponse         - liveness check (fast)
  ComponentStatus        - status per komponen
  HealthDetailedResponse - readiness check (detail)
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════════
# LIVENESS
# ══════════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    """Respon GET /v1/health (fast liveness check)."""

    status:    str = Field(..., description="'ok' | 'degraded' | 'down'")
    timestamp: str


# ══════════════════════════════════════════════════════════════════════════════
# READINESS
# ══════════════════════════════════════════════════════════════════════════════

class ComponentStatus(BaseModel):
    """Status per komponen (Redis, PostgreSQL, ChromaDB, dst)."""

    status:     str           = Field(..., description="'ok' | 'error'")
    latency_ms: Optional[int] = None
    detail:     Optional[str] = None


class HealthDetailedResponse(BaseModel):
    """Respon GET /v1/health/detailed (full readiness check)."""

    status:     str
    timestamp:  str
    components: Dict[str, ComponentStatus]