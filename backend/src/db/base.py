"""
SQLAlchemy Base — semua ORM model inherit dari sini.

Pakai DeclarativeBase (SQLAlchemy 2.0 style) supaya type-safe dan kompatibel
dengan Mapped[]/mapped_column() pada model.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class untuk semua SQLAlchemy ORM model."""
    pass