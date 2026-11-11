"""Engine + session factory. SQLite foreign keys are OFF by default, so we turn them on."""
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

import config


@event.listens_for(Engine, "connect")
def _enable_sqlite_fk(dbapi_conn, _):
    if dbapi_conn.__class__.__module__.startswith("sqlite3"):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


def make_engine(url: str | None = None):
    return create_engine(url or config.DATABASE_URL, future=True)


engine = make_engine()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def get_session():
    return SessionLocal()
