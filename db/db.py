import warnings
warnings.filterwarnings("ignore", message=".*pandas only supports SQLAlchemy.*")

import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import execute_values
import pandas as pd
from contextlib import contextmanager
from config import DATABASE_URL


# ---------------------------------------------------------------------------
# Connection pool (1-5 connections, reused across all queries)
# ---------------------------------------------------------------------------
_pool = None


def _get_pool():
    global _pool
    if _pool is None or _pool.closed:
        _pool = SimpleConnectionPool(1, 5, DATABASE_URL)
    return _pool


@contextmanager
def get_conn():
    """Get a connection from the pool. Auto-returns on exit."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


def query(sql, params=None):
    """Run a SELECT and return a DataFrame."""
    with get_conn() as conn:
        return pd.read_sql(sql, conn, params=params)


def execute(sql, params=None):
    """Run an INSERT/UPDATE/DELETE."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()


def bulk_insert(table, columns, data):
    """
    Bulk insert rows using execute_values (fast).

    Args:
        table: table name string
        columns: list of column name strings
        data: list of tuples, one per row
    """
    cols = ", ".join(columns)
    sql = f"INSERT INTO {table} ({cols}) VALUES %s ON CONFLICT DO NOTHING"
    with get_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, data, page_size=1000)
        conn.commit()
    print(f"Inserted {len(data)} rows into {table}")
