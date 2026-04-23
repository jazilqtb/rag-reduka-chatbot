"""
Router aggregator — mount semua endpoint di sini.
Prefix /v1 ditambahkan di main.py sehingga semua route = /v1/<prefix_router>.

Route summary:
  /v1/chat                          → chat.py
  /v1/documents/...                 → document.py
  /v1/session/...                   → session.py
  /v1/health, /v1/health/detailed   → health.py
"""

from fastapi import APIRouter

from src.api.endpoints import chat, document, session, health

api_router = APIRouter()

api_router.include_router(chat.router)
api_router.include_router(document.router)
api_router.include_router(session.router)
api_router.include_router(health.router)