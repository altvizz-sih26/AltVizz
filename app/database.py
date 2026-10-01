"""
Database configuration.

Uses SQLite for local dev (zero setup). To switch to PostgreSQL later,
just change DATABASE_URL to something like:
    postgresql://user:password@localhost:5432/depthwizard
No other code needs to change.
"""
import os
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./depthwizard.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def migrate_schema():
    """Add columns introduced after the initial SQLite dev schema."""
    additions = {
        "jobs": {
            "after_upload_id": "VARCHAR REFERENCES uploads(id)",
            "kind": "VARCHAR NOT NULL DEFAULT 'single'",
            "stage": "VARCHAR",
            "progress": "INTEGER NOT NULL DEFAULT 0",
        },
        "results": {
            "dsm_geotiff_path": "VARCHAR",
            "before_dsm_geotiff_path": "VARCHAR",
            "after_dsm_geotiff_path": "VARCHAR",
            "analysis_path": "VARCHAR",
            "before_flythrough_path": "VARCHAR",
            "change_summary": "TEXT",
        },
    }

    with engine.begin() as connection:
        inspector = inspect(connection)
        tables = set(inspector.get_table_names())
        for table, columns in additions.items():
            if table not in tables:
                continue
            existing = {column["name"] for column in inspect(connection).get_columns(table)}
            for column, definition in columns.items():
                if column not in existing:
                    connection.execute(text(
                        f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
                    ))
                    existing.add(column)


def get_db():
    """FastAPI dependency that yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
