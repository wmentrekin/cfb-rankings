import json
import logging
import os
from typing import Optional

import boto3  # type: ignore
from dotenv import load_dotenv  # type: ignore
from sqlalchemy import create_engine  # type: ignore

from artifacts.rankings import (
    build_payload,
    compute_rank_and_delta,
    get_ratings_with_conference,
    previous_week_with_data,
)

# Reuse the same logger name main.py configures via utils.setup_logging, so warnings from this
# module surface through the pipeline's existing stdout/file handlers when run via main.py, and
# still fall back to Python's default stderr handler when this module is used standalone.
logger = logging.getLogger("cfb_lp")


def get_r2_client():
    """
    Build a boto3 client for Cloudflare R2 from env vars.
    Returns:
        Optional[boto3 client]: A configured S3-compatible client, or None if credentials are
                                 missing/invalid. Never raises.
    """
    load_dotenv()
    account_id = os.getenv("R2_ACCOUNT_ID")
    access_key_id = os.getenv("R2_ACCESS_KEY_ID")
    secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")

    if not account_id or not access_key_id or not secret_access_key:
        logger.warning("R2 credentials not fully configured (R2_ACCOUNT_ID/R2_ACCESS_KEY_ID/"
                        "R2_SECRET_ACCESS_KEY); artifact publishing will be skipped.")
        return None

    try:
        client = boto3.client(
            service_name="s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name="auto",
        )
    except Exception as e:
        logger.warning("Failed to construct R2 client: %s", e)
        return None

    return client


def upload_json(client, bucket, key, payload) -> bool:
    """
    Upload a JSON-serializable payload to R2 under the given key.
    Args:
        client: boto3 S3-compatible client (as returned by get_r2_client).
        bucket (str): R2 bucket name.
        key (str): Object key to upload to.
        payload: JSON-serializable object.
    Returns:
        bool: True on success, False on any failure. Never raises.
    """
    try:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(payload, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        return True
    except Exception as e:
        logger.warning("Failed to upload rankings artifact to R2 key %s: %s", key, e)
        return False


def publish_rankings_artifact(year, week) -> None:
    """
    Read ratings back out of Postgres, compute rank/delta, and publish the rankings artifact
    for this season/week to R2 under both a historical and a latest key.
    Args:
        year (int): Season year.
        week (int): Week number.
    Returns:
        None
    Notes:
        Fully exception-safe: any failure (DB read, R2 credentials, upload) is logged as a
        warning and this function returns cleanly. It must never raise, since the ratings DB
        write has already succeeded by the time this runs and must not be put at risk.
    """
    try:
        load_dotenv()
        db_url = (
            f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
            f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
            "?sslmode=require"
        )
        engine = create_engine(db_url)
        try:
            current_df = get_ratings_with_conference(engine, year, week)
            prev_week: Optional[int] = previous_week_with_data(engine, year, week)
            previous_df = get_ratings_with_conference(engine, year, prev_week) if prev_week is not None else None
        finally:
            engine.dispose()

        ranked_df = compute_rank_and_delta(current_df, previous_df)
        payload = build_payload(year, week, ranked_df)

        client = get_r2_client()
        if client is None:
            logger.warning("R2 not configured, skipping artifact publish")
            return

        bucket = os.getenv("R2_BUCKET_NAME")
        if not bucket:
            logger.warning("R2_BUCKET_NAME not configured, skipping artifact publish")
            return

        keys = [
            f"rankings/{year}/week-{week:02d}.json",
            f"rankings/{year}/latest.json",
            "rankings/latest.json",
        ]
        for key in keys:
            if upload_json(client, bucket, key, payload):
                logger.info("Published rankings artifact to R2 key %s", key)
            else:
                logger.warning("Rankings artifact publish failed for R2 key %s", key)
    except Exception as e:
        logger.exception("publish_rankings_artifact failed unexpectedly for year=%s week=%s: %s", year, week, e)
        return
