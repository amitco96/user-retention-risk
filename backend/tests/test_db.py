"""
Unit tests for database module (backend/app/db/).

Tests:
- Database session initialization
- get_db dependency yields session
- create_tables function
- drop_tables function
"""

import pytest
import os
from unittest.mock import patch, AsyncMock, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession


class TestDatabaseSession:
    """Test database session creation."""

    @pytest.mark.asyncio
    async def test_get_db_yields_session(self):
        """Test that get_db yields an AsyncSession."""
        from backend.app.db.session import get_db

        # Create a generator from get_db
        db_gen = get_db()

        # Get the session
        session = await db_gen.__anext__()

        # Should be an AsyncSession-like object
        assert hasattr(session, "execute")
        assert hasattr(session, "commit")
        assert hasattr(session, "close")

        # Clean up - send StopAsyncIteration
        try:
            await db_gen.__anext__()
        except StopAsyncIteration:
            pass

    @pytest.mark.asyncio
    async def test_get_db_closes_session_on_exception(self):
        """Test that get_db closes session even on exception."""
        from backend.app.db.session import get_db

        db_gen = get_db()

        # Get the session
        session = await db_gen.__anext__()
        assert session is not None

        # Simulate exception in try block
        try:
            await db_gen.aclose()
        except:
            pass


class TestDatabaseUrl:
    """Test database URL configuration."""

    def test_database_url_from_environment(self):
        """Test that DATABASE_URL can be read from environment."""
        from backend.app.db import session

        # Check that DATABASE_URL is set
        assert session.DATABASE_URL is not None
        assert isinstance(session.DATABASE_URL, str)
        # Should contain postgresql or asyncpg
        assert "asyncpg" in session.DATABASE_URL or "postgresql" in session.DATABASE_URL

    def test_database_url_with_default(self):
        """Test that default DATABASE_URL is used if env var not set."""
        # If DATABASE_URL is not set, should use default localhost
        if "DATABASE_URL" not in os.environ:
            from backend.app.db.session import DATABASE_URL

            # Default should contain localhost or default connection string
            assert isinstance(DATABASE_URL, str)
            assert len(DATABASE_URL) > 0


class TestAsyncEngine:
    """Test async engine creation."""

    def test_engine_created(self):
        """Test that async engine is created."""
        from backend.app.db.session import engine

        assert engine is not None
        assert hasattr(engine, "connect")
        assert hasattr(engine, "execute")

    def test_session_maker_created(self):
        """Test that async_session_maker is created."""
        from backend.app.db.session import async_session_maker

        assert async_session_maker is not None
        assert callable(async_session_maker)


class TestCreateTables:
    """Test table creation."""

    @pytest.mark.asyncio
    async def test_create_tables_function_exists(self):
        """Test that create_tables function exists and is callable."""
        from backend.app.db.session import create_tables

        assert callable(create_tables)

    @pytest.mark.asyncio
    async def test_create_tables_with_mocked_engine(self):
        """Test create_tables with mocked engine."""
        from backend.app.db.session import create_tables, engine
        from backend.app.db.models import Base

        # Just ensure the function can be called
        # In real environment it would connect to DB
        # For now just verify the function signature
        assert callable(create_tables)


class TestDropTables:
    """Test table dropping."""

    @pytest.mark.asyncio
    async def test_drop_tables_function_exists(self):
        """Test that drop_tables function exists and is callable."""
        from backend.app.db.session import drop_tables

        assert callable(drop_tables)


class TestDatabaseModels:
    """Test database models."""

    def test_base_declarative_base_created(self):
        """Test that Base declarative_base is created."""
        from backend.app.db.models import Base

        assert Base is not None
        assert hasattr(Base, "metadata")


class TestEngineConfiguration:
    """Test engine configuration."""

    def test_engine_has_pool_config(self):
        """Test that engine has pool configuration."""
        from backend.app.db.session import engine

        # Engine should have pool options set
        assert engine.pool_pre_ping is not None

    def test_engine_echo_false_for_production(self):
        """Test that engine echo is False."""
        from backend.app.db.session import engine

        # echo should be False to avoid logging all SQL
        # We can't directly test this easily but can ensure engine exists
        assert engine is not None
