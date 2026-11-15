"""Security middleware and header validation for tier capping and mitigation modes.

Implements REQ-GW-05, REQ-RET-06, NFR-SEC-01, ADR-0004.
Traceability: UT_GW_05, ST_SEC_01.
"""

import logging
import os
from typing import Any, Dict, Mapping, Optional, Tuple
from fastapi import HTTPException, status

from src.schemas.verification import MitigationMode, VerificationRequest

logger = logging.getLogger("gateway.security")

HEADER_MAX_TIER = "x-max-tier"
HEADER_MITIGATION_MODE = "x-mitigation-mode"


def get_server_max_tier() -> int:
    """Read immutable server max tier ceiling from environment (default: 3)."""
    env_tier = os.environ.get("SERVER_MAX_TIER")
    if env_tier is not None:
        try:
            return int(env_tier)
        except ValueError:
            return 3
    return 3


def parse_security_headers(headers: Mapping[str, str]) -> Dict[str, Any]:
    """Extract and validate caller-specified security headers (REQ-GW-05).

    Headers:
        X-Max-Tier: 1, 2, or 3.
        X-Mitigation-Mode: flag, rewrite, redact, or block.

    Returns:
        Dict with parsed optional overrides: 'max_retrieval_tier' and 'mitigation_mode'.

    Raises:
        HTTPException: 422 Unprocessable Entity if header values are invalid.
    """
    overrides: Dict[str, Any] = {}

    raw_max_tier: Optional[str] = None
    raw_mit_mode: Optional[str] = None

    for k, v in headers.items():
        k_lower = k.lower()
        if k_lower in (HEADER_MAX_TIER, "x-dehallucination-max-tier"):
            raw_max_tier = v
        elif k_lower in (HEADER_MITIGATION_MODE, "x-dehallucination-mitigation"):
            raw_mit_mode = v

    if raw_max_tier is not None:
        try:
            tier_val = int(raw_max_tier.strip())
            if tier_val not in (1, 2, 3):
                raise ValueError("Out of range")
            overrides["max_retrieval_tier"] = tier_val
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid {HEADER_MAX_TIER} header: '{raw_max_tier}'. Must be 1, 2, or 3.",
            ) from exc

    if raw_mit_mode is not None:
        mode_str = raw_mit_mode.strip().lower()
        try:
            overrides["mitigation_mode"] = MitigationMode(mode_str)
        except ValueError as exc:
            valid_modes = [m.value for m in MitigationMode]
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Invalid {HEADER_MITIGATION_MODE} header: '{raw_mit_mode}'. "
                    f"Must be one of {valid_modes}."
                ),
            ) from exc

    return overrides


def apply_header_overrides(
    request: VerificationRequest,
    headers: Mapping[str, str],
) -> VerificationRequest:
    """Apply caller-specified headers to a VerificationRequest payload.

    Implements REQ-GW-05.
    """
    overrides = parse_security_headers(headers)
    if not overrides:
        return request

    update_dict: Dict[str, Any] = {}
    if "max_retrieval_tier" in overrides:
        update_dict["max_retrieval_tier"] = overrides["max_retrieval_tier"]
    if "mitigation_mode" in overrides:
        update_dict["mitigation_mode"] = overrides["mitigation_mode"]

    return request.model_copy(update=update_dict)


def enforce_server_tier_ceiling(
    client_tier: int,
    server_ceiling: Optional[int] = None,
) -> Tuple[int, bool]:
    """Enforce server max tier ceiling on client tier request (REQ-RET-06).

    Args:
        client_tier: Tier requested by client (1, 2, or 3).
        server_ceiling: Server-side ceiling (defaults to get_server_max_tier()).

    Returns:
        Tuple of (effective_tier, was_capped).
    """
    ceiling = server_ceiling if server_ceiling is not None else get_server_max_tier()
    if client_tier > ceiling:
        logger.warning(
            "Client requested tier %d exceeding server ceiling %d; capping at %d",
            client_tier,
            ceiling,
            ceiling,
        )
        return ceiling, True
    return client_tier, False
