"""Autenticación por API key (header X-API-Key).

Implementado como dependencia de FastAPI y no como middleware ASGI global:
  - aplica sólo a las rutas que lo declaran (/health queda público)
  - devuelve el client_id tipado (lo consume el rate limit)
  - se testea/overridea con `app.dependency_overrides`
"""

import hmac

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _lookup(keys: dict[str, str], candidate: str) -> str | None:
    # Comparación en tiempo constante para no filtrar info por timing.
    for key, client_id in keys.items():
        if hmac.compare_digest(key.encode(), candidate.encode()):
            return client_id
    return None


async def require_api_key(request: Request, api_key: str | None = Security(api_key_header)) -> str:
    if not api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="missing_api_key")
    client_id = _lookup(request.app.state.settings.api_key_map, api_key)
    if client_id is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="invalid_api_key")
    return client_id
