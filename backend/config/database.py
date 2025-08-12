"""Database configuration and initialization."""

import asyncio
from typing import AsyncGenerator, Optional

from loguru import logger
from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .settings import get_settings

settings = get_settings()


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass


# Database engines
engine: Optional[object] = None
async_engine: Optional[object] = None
SessionLocal: Optional[sessionmaker] = None
AsyncSessionLocal: Optional[async_sessionmaker] = None


async def init_database() -> None:
    """Initialize database connections."""
    global engine, async_engine, SessionLocal, AsyncSessionLocal
    
    try:
        logger.info("Initializing database connections")
        
        # Use AlloyDB in production if configured, otherwise SQLite for demo
        if settings.is_production and settings.alloydb_url:
            database_url = settings.alloydb_url
            logger.info(f"Using AlloyDB: {settings.ALLOYDB_HOST}:{settings.ALLOYDB_PORT}")
        else:
            database_url = settings.DATABASE_URL
            logger.info(f"Using SQLite for demo: {database_url}")
        
        # Create async engine
        async_engine = create_async_engine(
            database_url,
            echo=settings.is_development,
            future=True,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        
        # Create async session factory
        AsyncSessionLocal = async_sessionmaker(
            bind=async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False
        )
        
        # For SQLite, enable foreign key constraints
        if "sqlite" in database_url.lower():
            @event.listens_for(async_engine.sync_engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=10000")
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.close()
        
        # Create tables
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database initialization complete")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


async def get_database() -> AsyncGenerator[AsyncSession, None]:
    """Get database session."""
    if not AsyncSessionLocal:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()


async def close_database() -> None:
    """Close database connections."""
    global async_engine
    
    if async_engine:
        await async_engine.dispose()
        logger.info("Database connections closed")


# Health check function
async def check_database_health() -> bool:
    """Check database connectivity."""
    try:
        async with get_database() as session:
            await session.execute("SELECT 1")
            return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
