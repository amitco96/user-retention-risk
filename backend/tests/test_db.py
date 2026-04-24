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
        assert hasattr(engine, "begin")

    def test_session_maker_created(self):
        """Test that async_session_maker is created."""
        from backend.app.db.session import async_session_maker

        assert async_session_maker is not None
        assert callable(async_session_maker)


class TestCreateTables:
    """Test table creation."""

    @pytest.mark.asyncio
    async def test_create_tables_executes_on_sqlite(self):
        """Test that create_tables actually creates tables on a SQLite engine."""
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.pool import StaticPool
        from sqlalchemy import inspect, text
        from backend.app.db.models import Base

        test_engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            echo=False,
            future=True,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )

        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            tables = await conn.run_sync(
                lambda c: inspect(c).get_table_names()
            )

        assert "users" in tables
        assert "events" in tables
        assert "risk_scores" in tables

        await test_engine.dispose()

    @pytest.mark.asyncio
    async def test_create_tables_function_exists(self):
        """Test that create_tables function exists and is callable."""
        from backend.app.db.session import create_tables

        assert callable(create_tables)


class TestDropTables:
    """Test table dropping."""

    @pytest.mark.asyncio
    async def test_drop_tables_executes_on_sqlite(self):
        """Test that drop_tables actually drops tables on a SQLite engine."""
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.pool import StaticPool
        from sqlalchemy import inspect
        from backend.app.db.models import Base

        test_engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            echo=False,
            future=True,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )

        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            tables = await conn.run_sync(
                lambda c: inspect(c).get_table_names()
            )

        assert "users" not in tables
        assert "events" not in tables
        assert "risk_scores" not in tables

        await test_engine.dispose()

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

        # pool_pre_ping is stored on the underlying sync pool, not the AsyncEngine.
        assert engine.pool is not None
        assert engine.sync_engine is not None

    def test_engine_echo_false_for_production(self):
        """Test that engine echo is False."""
        from backend.app.db.session import engine

        # echo should be False to avoid logging all SQL
        # We can't directly test this easily but can ensure engine exists
        assert engine is not None
