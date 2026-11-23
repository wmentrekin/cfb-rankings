"""
One-off: build (and optionally publish) one season's playoffs artifact, and do nothing else.

Modelled on scripts/publish_season_schedule_artifact.py. `--season N` builds
playoffs/{N}/latest.json from the database; only with `--publish` is it uploaded to R2, after
which the object is read back and diffed against what this run built (ignoring generated_at_utc),
because publish_playoffs_artifact() never raises and so cannot by itself tell a real publish from a
skipped one. Only seasons configured in artifacts/playoffs.py's SEASON_RULES (2025, 2026) are
accepted; anything else exits non-zero before any DB/R2 work.

BLAST RADIUS
------------
Default (no `--publish`): read-only against the database; writes nothing except the optional
`--out` file. `--publish`: overwrites R2 keys `playoffs/{season}/latest.json` and
`playoffs/index.json` in place (atomic per-object PUTs), plus one read-only get_object to verify.

CREDENTIALS REQUIRED
--------------------
  default:    DB_HOST / DB_PORT / DB_USER / DB_PASSWORD / DB_NAME
  --publish:  the same, plus R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY / R2_BUCKET_NAME

Usage:
    python scripts/publish_playoffs_artifact.py --season 2026 --out payload.json
    python scripts/publish_playoffs_artifact.py --season 2026 --out payload.json --publish
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Running this file directly only puts scripts/ on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from artifacts.playoffs import SEASON_RULES, load_playoffs_payload, publish_playoffs_artifact  # noqa: E402


def _validate_season(season: int) -> None:
    """Exits non-zero for a season with no playoff rules, before any DB/R2 work runs."""
    if season not in SEASON_RULES:
        print(
            f"FAIL: --season {season} has no playoff rules (supported: {sorted(SEASON_RULES)}); refusing to run.",
            file=sys.stderr,
        )
        sys.exit(1)


def _validate_payload(payload) -> None:
    """Exits non-zero when nothing could be built or the bracket is empty."""
    if not payload or not payload.get("teams") or len(payload.get("games", [])) != 11:
        print("FAIL: no complete playoffs payload could be built; refusing to publish.", file=sys.stderr)
        sys.exit(1)


def _verify_published(season: int, expected_payload: dict) -> None:
    """Read playoffs/{season}/latest.json back from R2 and compare it with the payload this run
    built (ignoring generated_at_utc). Exits non-zero on any mismatch or read failure."""
    from artifacts.r2 import get_r2_client  # local import: only the --publish path needs it

    client = get_r2_client()
    if client is None:
        print("FAIL: R2 not configured; cannot verify the publish round-tripped.", file=sys.stderr)
        sys.exit(1)
    bucket = os.getenv("R2_BUCKET_NAME")
    if not bucket:
        print("FAIL: R2_BUCKET_NAME not set; cannot verify the publish round-tripped.", file=sys.stderr)
        sys.exit(1)

    key = f"playoffs/{season}/latest.json"
    try:
        published = json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())
    except Exception as e:
        print(f"FAIL: could not read back {key} from R2 after publish: {e}", file=sys.stderr)
        sys.exit(1)

    expected = {k: v for k, v in expected_payload.items() if k != "generated_at_utc"}
    got = {k: v for k, v in published.items() if k != "generated_at_utc"}
    if expected != got:
        print(
            f"FAIL: published {key} does not match the payload this run built "
            f"(ignoring generated_at_utc) -- the publish did not round-trip.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"VERIFIED: {key} round-tripped from R2 and matches the payload this run built.")


def _print_summary(payload: dict) -> None:
    print(f"season: {payload.get('season')}")
    print(f"kind: {payload.get('kind')}")
    print(f"teams: {len(payload.get('teams', []))}")
    print(f"games: {len(payload.get('games', []))}")
    print(f"first_out: {len(payload.get('first_out', []))}")


def main() -> None:
    # INFO so publish_playoffs_artifact's "Published playoffs artifact to R2 key ..." line (the
    # one the workflow greps for) reaches stdout when run standalone.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, required=True, help="Season year to publish, e.g. 2026.")
    parser.add_argument(
        "--publish", action="store_true",
        help="Actually publish to R2 and verify the round trip. Without it nothing is uploaded.",
    )
    parser.add_argument("--out", type=str, default=None, help="Optional path to write the built payload as JSON.")
    args = parser.parse_args()

    _validate_season(args.season)
    payload = load_playoffs_payload(args.season)

    if args.out and payload is not None:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote payload to {args.out}")

    _validate_payload(payload)

    if not args.publish:
        print(f"DRY RUN -- season={args.season}: payload built{' and written' if args.out else ''}; "
              f"nothing uploaded to R2. Pass --publish to actually publish.")
    else:
        print(f"Publishing season={args.season} to R2...")
        publish_playoffs_artifact(args.season)
        print(f"publish_playoffs_artifact({args.season}) returned; verifying the round trip...")
        _verify_published(args.season, payload)

    _print_summary(payload)


if __name__ == "__main__":
    main()
