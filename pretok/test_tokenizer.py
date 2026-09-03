"""Tokenizer roundtrip verification test.

Fetches Parquet chunks from the HuggingFace Hub dataset:
    https://huggingface.co/datasets/leonidas123/valkmodel-data

For each token-length bucket (1024, 2048, 4096, 8192, 16384):
  1. Downloads the first parquet file from the repo.
  2. Filters rows whose n_tokens falls within ±10% of the bucket target.
  3. Samples 5 random rows from the filtered set.
  4. Decodes stored token IDs with MathTokenizer.
  5. Appends one line per row to `results.txt`:
         [seq=BUCKET] row=R | source=S | n_tokens=N | decoded="..." | PASS|FAIL

A row is PASS if:
  - tokenizer.decode(tokens) returns non-empty text
  - len(tokens) == stored n_tokens

Usage:
  python pretok/test_tokenizer.py

NOTE: Run `python -m pretok.pretokenize` first to upload the pretokenized data.
"""

from __future__ import annotations

import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

HF_REPO_ID = "leonidas123/valkmodel-data"
RESULTS_FILE = Path(__file__).parent / "results.txt"

# Token-length buckets and the acceptance window (±10% of target).
SEQ_LEN_BUCKETS = [1024, 2048, 4096, 8192, 16384]
BUCKET_TOLERANCE = 0.10  # ±10%
ROWS_PER_BUCKET = 5


def _get_hf_token() -> str:
    """Read HF token from environment or .env file."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                t = line.split("=", 1)[1].strip().strip('"').strip("'")
                if t:
                    return t
    raise RuntimeError(
        "HF_TOKEN not set in environment or .env. "
        "Cannot download verification data from HuggingFace Hub."
    )


def build_tokenizer():
    """Build and return the MathTokenizer (imports at call site)."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.tokenizer.math_tokenizer import MathTokenizer
    return MathTokenizer()


def _list_train_parquet_files(repo_id: str, token: str) -> List[str]:
    """Return all parquet file paths under data/train/ in the repo."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception as exc:
        raise RuntimeError(f"Cannot list files in '{repo_id}': {exc}") from exc

    return sorted(
        f for f in all_files
        if f.startswith("data/train/") and f.endswith(".parquet")
    )


def _download_parquet(
    repo_id: str,
    file_path: str,
    token: str,
) -> Tuple[List[Dict[str, Any]], int]:
    """Download a parquet file and return (list of rows, total_row_count).

    Downloads to a temp file via hf_hub_download, then reads with pyarrow.
    Returns the full table as a list of dicts and the total row count from metadata.
    """
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=file_path,
        repo_type="dataset",
        token=token,
    )
    pf = pq.ParquetFile(local_path)
    total_rows = pf.metadata.num_rows
    table = pf.read()
    records = table.to_pydict()

    # Convert pyarrow ChunkedArrays to plain Python lists so [] indexing works
    records = {k: list(v) for k, v in records.items()}

    rows: List[Dict[str, Any]] = []
    n = len(next(iter(records.values())))
    for i in range(n):
        rows.append({
            "source":   records["source"][i],
            "text":     records["text"][i],
            "tokens":   records["tokens"][i],
            "n_tokens": records["n_tokens"][i],
        })
    return rows, total_rows


def _bucket_key(bucket: int) -> Tuple[int, int]:
    """Return (lo, hi) token-count range for a bucket."""
    lo = int(bucket * (1 - BUCKET_TOLERANCE))
    hi = int(bucket * (1 + BUCKET_TOLERANCE))
    return lo, hi


def _roundtrip_row(row: Dict[str, Any], tokenizer) -> Tuple[bool, str]:
    """Decode a row's tokens and return (ok, display_string)."""
    tokens = list(row["tokens"])  # ensure list, not array
    stored_n = int(row["n_tokens"])

    try:
        decoded = tokenizer.decode(tokens)
    except Exception as exc:
        return False, f"DECODE_ERROR: {exc}"

    ok = bool(decoded.strip()) and len(tokens) == stored_n
    # Truncate for log readability
    display = decoded[:120].replace("\n", "\\n")
    return ok, display


def run_test() -> None:
    """Run the full verification pipeline."""
    t_start = time.time()
    print("=== Tokenizer roundtrip verification ===\n")

    hf_token = _get_hf_token()
    tokenizer = build_tokenizer()
    print(f"Tokenizer vocab size : {tokenizer.n_vocab}\n")

    # ── Find available parquet files in the repo ──────────────────────────
    print("[hf] Listing files in repository ...")
    try:
        train_files = _list_train_parquet_files(HF_REPO_ID, hf_token)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        print(
            "\nNOTE: The dataset may not exist yet. "
            "Run `python -m pretok.pretokenize` first to upload pretokenized data."
        )
        sys.exit(1)

    if not train_files:
        print(
            f"ERROR: No parquet files found in {HF_REPO_ID}/data/train/. "
            "Run pretokenization first."
        )
        sys.exit(1)

    print(f"Found {len(train_files)} parquet file(s): {train_files}\n")

    # ── Download the first parquet file ─────────────────────────────────────
    chosen_file = train_files[0]
    print(f"[download] Fetching {chosen_file} ...")
    try:
        all_rows, total_rows = _download_parquet(
            HF_REPO_ID, chosen_file, hf_token
        )
    except Exception as exc:
        print(f"ERROR downloading '{chosen_file}': {exc}")
        sys.exit(1)

    print(f"[download] Read {len(all_rows)} rows from {total_rows:,} total.\n")

    # ── Per-bucket filtering and roundtrip test ────────────────────────────
    rng = random.Random(42)  # reproducible sampling
    result_lines: List[str] = []
    timestamp = datetime.utcnow().isoformat()

    result_lines.append(
        f"# Tokenizer roundtrip verification — {timestamp}"
    )
    result_lines.append(
        f"# HF repo   : https://huggingface.co/datasets/{HF_REPO_ID}"
    )
    result_lines.append(
        f"# Source    : {chosen_file} ({total_rows:,} rows in file)"
    )
    result_lines.append(f"# Buckets   : {SEQ_LEN_BUCKETS}")
    result_lines.append(
        f"# Tolerance : ±{BUCKET_TOLERANCE*100:.0f}% per bucket"
    )
    result_lines.append(f"# Sample N  : {ROWS_PER_BUCKET} rows per bucket")
    result_lines.append("")

    overall_pass = 0
    overall_fail = 0

    for bucket in SEQ_LEN_BUCKETS:
        lo, hi = _bucket_key(bucket)

        # Filter rows that fall within this bucket
        bucket_rows = [
            (i, row) for i, row in enumerate(all_rows)
            if lo <= row["n_tokens"] <= hi
        ]

        if not bucket_rows:
            print(
                f"[seq={bucket:>5}] (lo={lo}, hi={hi}) "
                f"→ 0 rows found in this file | SKIP"
            )
            result_lines.append(
                f"[seq={bucket:>5}] No rows found in bucket range "
                f"[{lo}, {hi}] — SKIP"
            )
            result_lines.append("")
            continue

        print(
            f"[seq={bucket:>5}] (lo={lo}, hi={hi}) "
            f"→ {len(bucket_rows)} candidate rows"
        )

        # Sample up to ROWS_PER_BUCKET
        sampled = (
            bucket_rows
            if len(bucket_rows) <= ROWS_PER_BUCKET
            else rng.sample(bucket_rows, ROWS_PER_BUCKET)
        )

        bucket_pass = 0
        bucket_fail = 0
        for (row_idx, row) in sampled:
            ok, display = _roundtrip_row(row, tokenizer)
            status = "PASS" if ok else "FAIL"
            if ok:
                bucket_pass += 1
                overall_pass += 1
            else:
                bucket_fail += 1
                overall_fail += 1

            result_lines.append(
                f"[seq={bucket:>5}] row={row_idx} "
                f"| source={row['source']:<10} "
                f"| n_tokens={row['n_tokens']:>5} "
                f"| decoded=\"{display}...\" "
                f"| {status}"
            )

        print(
            f"            Sampled {len(sampled)} | "
            f"PASS={bucket_pass} FAIL={bucket_fail}"
        )
        result_lines.append("")

    # ── Write results to file ─────────────────────────────────────────────
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(result_lines) + "\n")

    elapsed = time.time() - t_start
    total_tested = overall_pass + overall_fail

    print(f"\n=== Done ({elapsed:.1f}s) ===")
    print(f"Rows tested  : {total_tested}")
    print(f"PASS         : {overall_pass}")
    print(f"FAIL         : {overall_fail}")
    print(f"Results file : {RESULTS_FILE}")

    if overall_fail > 0:
        print("\nWARNING: Some rows failed roundtrip. Check results.txt.")
        sys.exit(1)


if __name__ == "__main__":
    run_test()
