import os

from dotenv import load_dotenv  # type: ignore
from sqlalchemy import create_engine, text  # type: ignore


def ping_db():
    """
    Run a trivial query against Supabase so the project registers activity and
    doesn't get auto-paused after 7 days of inactivity.
    """
    load_dotenv()
    db_url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        "?sslmode=require"
    )
    engine = create_engine(db_url)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    engine.dispose()
    print("Supabase keepalive ping succeeded.")


if __name__ == "__main__":
    ping_db()
