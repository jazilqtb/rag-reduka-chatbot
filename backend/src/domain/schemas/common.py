"""
Schema umum yang dipakai di lebih dari satu endpoint.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════════
# ERROR
# ══════════════════════════════════════════════════════════════════════════════

class ErrorResponse(BaseModel):
    """Format error standar untuk semua endpoint."""

    error:   str = Field(..., description="Kode error singkat. Contoh: 'validation_error'")
    message: str = Field(..., description="Penjelasan human-readable.")
    detail:  Optional[Any] = Field(
        default=None,
        description="Detail tambahan jika ada (misal: field yang gagal validasi).",
    )