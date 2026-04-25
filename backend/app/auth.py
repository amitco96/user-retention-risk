"""
Bearer-token auth for internal pipeline endpoints.

The token is read from the API_PIPELINE_SECRET environment variable at request
time (not import time) so tests can use monkeypatch.setenv().
"""

import os
from fastapi import Header, HTTPException, status


def verify_pipeline_token(authorization: str = Header(None)) -> None:
    """
    FastAPI dependency. Raises 401 unless the request carries
    `Authorization: Bearer <API_PIPELINE_SECRET>`.

    Fails closed: if API_PIPELINE_SECRET is unset, every request is rejected.
    """
    secret = os.environ.get("API_PIPELINE_SECRET")
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Pipeline auth not configured",
        )
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed bearer token",
        )
    token = authorization[len("Bearer "):].strip()
    if token != secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
        )
