import structlog
from fastapi import HTTPException, Header, Request

from gateway.config import TenantConfig

logger = structlog.get_logger()


async def get_current_tenant(
    request: Request,
    authorization: str | None = Header(None),
) -> TenantConfig:
    """Extract and validate Bearer token from Authorization header.

    Returns the TenantConfig for the authenticated tenant.
    Raises 401 if the token is missing, malformed, or invalid.
    """
    if authorization is None or not authorization.startswith("Bearer "):
        logger.warning("auth_failed", reason="missing_or_invalid_header")
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header",
        )

    token = authorization[7:]  # Strip "Bearer "
    if not token:
        logger.warning("auth_failed", reason="empty_token")
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    registry = request.app.state.registry
    tenant = registry.api_key_to_tenant.get(token)
    if tenant is None:
        logger.warning("auth_failed", reason="invalid_key")
        raise HTTPException(status_code=401, detail="Invalid API key")

    logger.info("auth_success", tenant_id=tenant.id)
    return tenant
