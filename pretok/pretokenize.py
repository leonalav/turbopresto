"""Pretokenization pipeline for RWKV-7 math model (50M parameters).

Fetches data from:
  - issdandavis/UltraData-Math  (L2-preview, forum math w/ LaTeX, 40% of corpus)
  - HuggingFaceFW/fineweb (10BT) (general English web text, 30% of corpus)
  - ajibawa-2023/Python-Code-Large (Python source files, Python 3 filtered, 30%)

────────────────────────────────────────────────────────────────────────────
Compute-optimal sizing (per Chinchilla / Hoffmann et al. 2022):
  Target tokens  = 40 × N_params ≈ 40 × 50M = 2.0 B tokens
  (Chinchilla is 20×; we use 40× because math/scientific corpora are denser
   and we want extra reasoning signal.)
────────────────────────────────────────────────────────────────────────────

Target split: 40% math / 30% general / 30% Python code
  - issdandavis/UltraData-Math → 40% × 2.0B = 800M tokens
  - FineWeb-10BT               → 30% × 2.0B = 600M tokens
  - ajibawa-2023/Python-Code...→ 30% × 2.0B = 600M tokens

UltraData-Math (L2-preview): 4 parquet shards are pre-downloaded to
~/.cache/resonanc/ultradata-math/ (~1.6 GB, ~20 min at 1.4 MB/s).
Rows are shuffled in-memory and quality-filtered (quality_label >= 3).
4 shards ≈ 400K rows ≈ 200M chars ≈ 40M tokens.

Python code is filtered to Python 3 syntax (rejects `print "foo"`,
`class X(object):`, `u'...'`, `apply()`, `exec '...'`).

────────────────────────────────────────────────────────────────────────────

NOTE: GSM8K and MATH (hendrycks/competition_math) are BENCHMARK datasets
used only for evaluation — they must NEVER appear in training data.

NOTE: open-r1/OpenR1-Math-220k and camel-ai/physics contain R1 reasoning
traces and physics dialogues respectively; they belong in SFT, not pretrain.

Tokenizes with MathTokenizer and saves as Parquet files (~512 MB each).
Uploads to:
  https://huggingface.co/datasets/leonidas123/valkmodel-data

Usage:
  # Linux VM (24 CPUs): tokenize locally with all 24 cores, no upload
  python -m pretok.pretokenize --workers 24 --no-upload

  # Linux VM: tokenize + upload to HF Hub
  HF_TOKEN=hf_xxx python -m pretok.pretokenize --workers 24

  # Local sanity check (small caps, sequential)
  python -m pretok.pretokenize --workers 1 --max-swallow 100 \
      --max-python-code 1000 --no-upload

  # Modal 32-CPU VM
  modal run pretok/pretokenize.py

HF_TOKEN: read from the environment variable (e.g. `export HF_TOKEN=hf_xxx`
on bash, `$env:HF_TOKEN='hf_xxx'` on PowerShell) or, as a fallback, from a
.env file at the repo root.  The token is ONLY needed for the upload step
— tokenization and parquet writing work without it (the three source
datasets — issdandavis/UltraData-Math, HuggingFaceFW/fineweb,
ajibawa-2023/Python-Code-Large — are public and streamed without auth).
If HF_TOKEN is unset (or --no-upload is passed), the run skips upload and
keeps parquet chunks in ./parquet_chunks/.

VM notes:
  • Code is platform-neutral; runs identically on Linux, macOS, Windows.
  • Disk: expect ~10–12 GB of parquet under ./parquet_chunks/ before upload.
  • Workers default to os.cpu_count(). On Linux the pool uses the 'fork'
    start method (cheap). tiktoken is fork-safe; each worker constructs
    its own MathTokenizer.
  • If the run is interrupted, re-running resumes cleanly: parquet_chunks/
    is overwritten, the script doesn't track per-chunk resumption (start
    over if interrupted mid-run).
  • IPv6 is force-disabled at import time.  Some Linux VM providers ship
    without IPv6; HF Hub's HTTP client otherwise retries forever with
    [Errno 97] "Address family not supported by protocol".  The patch is
    a no-op on hosts that have working IPv4 only.
"""

from __future__ import annotations

import json
import os
import random
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# IPv4-only DNS (must run before huggingface_hub / urllib3 imports)
# ---------------------------------------------------------------------------
def _force_ipv4() -> None:
    """Coerce socket.getaddrinfo() to return only IPv4 results.

    Some Linux VM providers ship without IPv6.  The HF Hub HTTP client
    (httpx + urllib3) tries AAAA records first; when AAAA returns
    "[Errno 97] Address family not supported by protocol" it retries
    with backoff and eventually gives up.  Filtering out non-AF_INET
    entries up front makes every downstream call IPv4-only.

    Idempotent: re-applying replaces the same module attribute.  No-op
    on hosts where IPv4 works (all results are AF_INET already).
    """
    _orig = socket.getaddrinfo

    def _patched(host, *args, **kwargs):
        try:
            results = _orig(host, *args, **kwargs)
        except socket.gaierror:
            raise
        v4 = [r for r in results if r[0] == socket.AF_INET]
        if v4:
            return v4
        # Fall back to original list; will raise a normal DNS error if
        # nothing is reachable.  Better than silently dropping all results.
        return results

    socket.getaddrinfo = _patched


_force_ipv4()
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    import modal
    is_modal = True
except ImportError:
    is_modal = False
    modal = None  # type: ignore


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_REPO_ID = "leonidas123/valkmodel-data"
HF_REPO_TYPE = "dataset"
LOCAL_DATA_DIR = Path("parquet_chunks")

ROWS_PER_CHUNK = 100_000
CHUNK_SIZE_ESTIMATE_BYTES = 512 * 1024 * 1024  # 512 MB

# ── Compute-optimal targets (see module docstring for derivation) ───────────
TARGET_TOTAL_TOKENS = 2_000_000_000        # 2.0B tokens
MATH_RATIO          = 0.40                 # tokyotech-llm/swallow-math
GENERAL_RATIO       = 0.30                 # FineWeb
PYTHON_CODE_RATIO   = 0.30                 # ajibawa-2023/Python-Code-Large
# Ratios must sum to 1.0
assert abs(MATH_RATIO + GENERAL_RATIO + PYTHON_CODE_RATIO - 1.0) < 1e-9

# ── Avg tokens per document (used for pre-allocation to avoid buffering) ───
# M2 fix: estimates match the actual MathTokenizer default (digit_split=False).
# Without digit splitting, numbers are tokenized natively by cl100k (1 token
# per integer of reasonable length).  If a caller enables digit_split=True
# these estimates must be re-measured.
AVG_TOKENS = {
    "ultradata_math": 4_000,   # ~500 chars/row, 5 chars/token ≈ 100 tokens/row
    "fineweb":           700,   # typical web paragraph
    "python_code":       700,   # one Python file snippet
    "synthetic":          20,   # arithmetic sentence
}


@dataclass
class TokenizeConfig:
    """Configuration for the pretokenization run."""
    # Source caps (in DOCUMENTS, hard upper bound)
    max_swallow:     int = 50_000
    max_python_code: int = 500_000   # Python-Code-Large has ~2M rows; cap at 500k
    max_synthetic: int = 2_000

    hf_token_env: str = "HF_TOKEN"
    output_dir:   Path = field(default_factory=lambda: LOCAL_DATA_DIR)
    rows_per_chunk: int = ROWS_PER_CHUNK
    chunk_size_estimate: int = CHUNK_SIZE_ESTIMATE_BYTES
    # Tokenization parallelism.  Set to 1 to disable multiprocessing.
    workers: int = field(default_factory=lambda: max(1, (os.cpu_count() or 1)))
    # Records per pool task — small batches keep cap-tracking responsive.
    encode_batch_size: int = 32
    # If True, skip the HF Hub upload step even if HF_TOKEN is set.
    no_upload: bool = False


# ---------------------------------------------------------------------------
# Text sources (each is a generator yielding {source, text})
# ---------------------------------------------------------------------------

def stream_synthetic_texts(size: int, seed: int) -> Iterator[Dict[str, str]]:
    """Generate synthetic arithmetic examples (always works, no download)."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data.synthetic import SyntheticMathDataset
    print(f"[synthetic] Generating {size} examples ...")
    ds = SyntheticMathDataset(size=size, max_digits=3, seed=seed)
    for i in range(len(ds)):
        yield {"source": "synthetic", "text": ds.format_for_pretrain(i)}
    print(f"[synthetic] Generated {size} examples.")


def _download_jsonl_resumable(
    url: str,
    dst: Path,
    *,
    max_retries: int = 12,
    chunk_size: int = 4 * 1024 * 1024,
) -> Path:
    """Download a large JSONL file from HF Hub with byte-level resume.

    Why: HF's CDN routinely truncates 10 GB single-stream downloads after a
    few MB ("peer closed connection without sending complete message body").
    `load_dataset(streaming=True)` and `hf_hub_download` both treat that as
    a fatal retry-bounded failure and re-download from byte 0.

    This helper uses HTTP `Range` requests so a truncation only costs the
    last few chunks, and backs off exponentially with jitter when the CDN
    throttles us.
    """
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    dst.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    # urllib3 retry handles transient HTTP-level errors (503, 504, etc.)
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=4)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # Resolve final URL (HF redirects to a CDN bucket)
    head = session.head(url, allow_redirects=True, timeout=30)
    head.raise_for_status()
    final_url = head.url
    total = int(head.headers.get("Content-Length", 0))

    # Existing partial file?
    pos = dst.stat().st_size if dst.exists() else 0
    if pos == total and total > 0:
        print(f"[download] {dst.name}: already complete ({total / 1e9:.2f} GB)")
        return dst

    if pos > 0 and total > 0:
        print(f"[download] {dst.name}: resuming at {pos / 1e9:.2f} GB "
              f"of {total / 1e9:.2f} GB")

    attempt = 0
    while attempt < max_retries:
        try:
            headers = {"Range": f"bytes={pos}-"} if pos > 0 else {}
            r = session.get(
                final_url, headers=headers, stream=True, timeout=60
            )
            r.raise_for_status()

            mode = "ab" if pos > 0 else "wb"
            t0 = time.time()
            with open(dst, mode) as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pos += len(chunk)
            elapsed = time.time() - t0
            rate = pos / elapsed / 1e6 if elapsed > 0 else 0
            # Server may have closed early without telling us; verify size.
            actual = dst.stat().st_size
            if total > 0 and actual < total:
                raise requests.exceptions.ContentDecodingError(
                    f"short read: got {actual / 1e6:.1f} MB, "
                    f"expected {total / 1e6:.1f} MB"
                )
            print(f"[download] {dst.name}: done in {elapsed:.1f}s "
                  f"({rate:.1f} MB/s)")
            return dst
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ContentDecodingError) as e:
            attempt += 1
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Failed to download {dst.name} after {max_retries} "
                    f"retries at {pos / 1e9:.2f} GB. Last error: {e}"
                ) from e
            # Cap exponential backoff at 60s, add jitter
            sleep = min(60.0, 2 ** attempt) + random.uniform(0, 2)
            print(f"[download] {dst.name}: stream cut at {pos / 1e9:.2f} GB "
                  f"({e.__class__.__name__}). Retrying in {sleep:.1f}s "
                  f"[{attempt}/{max_retries}]")
            time.sleep(sleep)

    raise RuntimeError(f"Unreachable: download loop exited for {dst.name}")


def _iter_jsonl_records(path: Path) -> Iterator[Dict[str, str]]:
    """Stream JSONL records from a fully-downloaded local file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ---------------------------------------------------------------------------
# UltraData-Math streamer (parquet-based, partial pre-download)
# ---------------------------------------------------------------------------

class UltraDataMathStreamer:
    """Stream rows from issdandavis/UltraData-Math (L2-preview) by downloading
    a configurable number of parquet shards to local disk and reading from them.

    Why parquet: the dataset ships as 138 parquet shards (~415 MB each).
    Downloading even 3-4 shards gives 300-400K rows = ~200M chars =
    ~40M tokens — more than enough for the math compute-optimal budget.
    HF's CDN is too slow for the full 128 GB; streaming via the rows API
    (max 100 rows/request) is too chatty (160K+ sequential HTTP calls).

    The tradeoff: ~20 min to download 4 shards at 1.4 MB/s, then instant
    iteration from local disk.  This is the same pattern as swallow-math.

    Config used: UltraData-Math-L2-preview
    Schema      : {content: string, quality_label: int}
    Source      : forum posts with LaTeX math (quality labels 1-5)
    """

    DATASET   = "issdandavis/UltraData-Math"
    CONFIG    = "UltraData-Math-L2-preview"
    # Parquet files within the config (138 total, each ~100K rows)
    # We download a prefix of this list — caller controls how many via
    # `n_shards_to_cache`.
    PARQUET_FILES: List[str] = [
        f"data/UltraData-Math-L2-preview/UltraData-Math-L2-part-{i:05d}-of-00138.parquet"
        for i in range(1, 139)
    ]

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        n_shards_to_cache: int = 4,
        quality_threshold: int = 3,
        seed: int = 42,
    ):
        """
        Args:
            cache_dir       : where to store downloaded parquet shards.
                              Defaults to ~/.cache/resonanc/ultradata-math/
            n_shards_to_cache: how many shards to pre-download.
                              4 shards ≈ 400K rows ≈ 200M chars ≈ 40M tokens.
                              Each shard is ~415 MB on disk.
            quality_threshold: minimum quality_label to include (1-5).
                              Default 3 filters out low-quality forum noise.
            seed            : RNG seed for shuffling the row order.
        """
        if cache_dir is None:
            cache_dir = Path(os.environ.get(
                "ULTRADATA_CACHE_DIR",
                str(Path.home() / ".cache" / "resonanc" / "ultradata-math")
            ))
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.n_shards = n_shards_to_cache
        self.quality_threshold = quality_threshold
        self._rng = random.Random(seed)

    # ── Public API ──────────────────────────────────────────────────────────

    def ensure_cached(self) -> None:
        """Download parquet shards if not already in cache_dir.

        Idempotent: skips shards that already exist and are the right size.
        Prints progress as each shard completes.
        """
        for i, remote_path in enumerate(self.PARQUET_FILES[:self.n_shards]):
            local_path = self.cache_dir / remote_path.replace("/", "_")
            expected_bytes = self._shard_size(i)

            # Skip if already downloaded (with basic size sanity check)
            if local_path.exists() and local_path.stat().st_size >= expected_bytes * 0.9:
                print(f"  [ultradata] shard {i+1}/{self.n_shards}: "
                      f"already cached ({local_path.stat().st_size / 1e6:.0f} MB)")
                continue

            url = (
                f"https://huggingface.co/datasets/{self.DATASET}/resolve/main/"
                f"{remote_path}"
            )
            print(f"  [ultradata] shard {i+1}/{self.n_shards}: "
                  f"downloading {remote_path} ...")
            _download_jsonl_resumable(url, local_path)
            actual = local_path.stat().st_size
            print(f"  [ultradata] shard {i+1}/{self.n_shards}: "
                  f"done ({actual / 1e6:.0f} MB)")

    def stream(self, max_docs: int | None = None) -> Iterator[Dict[str, str]]:
        """Yield rows from the cached parquet shards.

        Each shard is read into memory (pyarrow, ~400 MB peak), shuffled
        in-memory, and yielded until max_docs is hit.

        Yields {"source": "ultradata_math", "text": <content>}.
        """
        import pyarrow.parquet as pq

        all_rows: List[Dict] = []
        for i, remote_path in enumerate(self.PARQUET_FILES[:self.n_shards]):
            local_path = self.cache_dir / remote_path.replace("/", "_")
            if not local_path.exists():
                raise RuntimeError(
                    f"Shard {i+1} not found at {local_path}. "
                    "Call ensure_cached() first."
                )
            table = pq.read_table(str(local_path))
            df = table.to_pandas()
            for _, row in df.iterrows():
                all_rows.append({
                    "content": row["content"],
                    "quality_label": row["quality_label"],
                })

        # Shuffle once before yielding
        self._rng.shuffle(all_rows)

        yielded = 0
        for row in all_rows:
            if max_docs is not None and yielded >= max_docs:
                break
            # Quality filter: only include rows at or above threshold
            if row["quality_label"] < self.quality_threshold:
                continue
            yield {
                "source": "ultradata_math",
                "text": row["content"],
            }
            yielded += 1

    # ── Internal helpers ────────────────────────────────────────────────────

    @staticmethod
    def _shard_size(shard_idx: int) -> int:
        """Estimated uncompressed size of shard `shard_idx` in bytes.

        Based on the first shard (415 MB).  Later shards may be slightly
        smaller or larger depending on row density.
        """
        return 415_000_000  # ~415 MB

    @staticmethod
    def total_shards() -> int:
        return len(UltraDataMathStreamer.PARQUET_FILES)


def stream_ultradata_texts(max_rows: int) -> Iterator[Dict[str, str]]:
    """Stream math texts from issdandavis/UltraData-Math (L2-preview).

    Downloads 4 parquet shards to ~/.cache/resonanc/ultradata-math/
    (~1.6 GB, ~20 min at 1.4 MB/s) on first run, then streams from disk.
    Rows are shuffled in-memory and quality-filtered (quality_label >= 3).

    Yields {"source": "ultradata_math", "text": <content>}.
    """
    cache_dir = Path(os.environ.get(
        "ULTRADATA_CACHE_DIR",
        str(Path.home() / ".cache" / "resonanc" / "ultradata-math")
    ))
    cache_dir.mkdir(parents=True, exist_ok=True)

    streamer = UltraDataMathStreamer(
        cache_dir=cache_dir,
        n_shards_to_cache=4,
        quality_threshold=3,
        seed=42,
    )

    total_shards = streamer.total_shards()
    print(f"[ultradata] Caching {streamer.n_shards}/{total_shards} parquet shards "
          f"({streamer.n_shards * 415} MB) → {cache_dir}")
    streamer.ensure_cached()
    print(f"[ultradata] Streaming records (cap={max_rows:,} docs, quality>=3) ...")

    count = 0
    for record in streamer.stream(max_docs=max_rows):
        yield record
        count += 1
        if count >= max_rows:
            break
    print(f"[ultradata] Streamed {count:,} rows.")


def stream_swallow_texts(max_rows: int) -> Iterator[Dict[str, str]]:
    """Stream math problems from tokyotech-llm/swallow-math.

    The raw dataset ships as two ~10 GB JSONL files.  HF's CDN routinely
    truncates such large single-stream downloads.  We use a resumable
    HTTP Range downloader to local disk first, then stream from disk —
    no per-row network calls, no truncation thrash.

    Raises RuntimeError on failure (caller must handle).
    Yields {"source": "swallow", "text": <problem + solution>}.
    """
    from huggingface_hub import hf_hub_download

    cache_dir = Path(os.environ.get(
        "SWALLOW_CACHE_DIR",
        str(Path.home() / ".cache" / "resonanc" / "swallow-math")
    ))
    cache_dir.mkdir(parents=True, exist_ok=True)

    base = (
        "https://huggingface.co/datasets/tokyotech-llm/swallow-math/"
        "resolve/main"
    )
    shards = [
        ("train-00001-of-00002.jsonl", f"{base}/train-00001-of-00002.jsonl"),
        ("train-00002-of-00002.jsonl", f"{base}/train-00002-of-00002.jsonl"),
    ]

    print(f"[swallow] Resumable download into {cache_dir}")
    for filename, url in shards:
        dst = cache_dir / filename
        _download_jsonl_resumable(url, dst)

    print(f"[swallow] Streaming records (cap={max_rows:,} docs) ...")
    count = 0
    for filename, _ in shards:
        for row in _iter_jsonl_records(cache_dir / filename):
            problem = row.get("problem", "")
            solution = row.get("solution", "")
            if not problem or not solution:
                continue
            text = f"Problem: {problem}\n\nSolution:\n{solution}"
            yield {"source": "swallow", "text": text}
            count += 1
            if count >= max_rows:
                print(f"[swallow] Streamed {count:,} rows.")
                return
    print(f"[swallow] Streamed {count:,} rows.")


def stream_fineweb_texts(target_tokens: int) -> Iterator[Dict[str, str]]:
    """Stream general web text from HuggingFaceFW/fineweb (sample/10BT).

    Stop after target_tokens are streamed.  We use `take(N)` filtering
    based on a rough token estimate per document.

    Raises RuntimeError on failure.
    """
    from datasets import load_dataset

    # Each row ≈ AVG_TOKENS["fineweb"] tokens.  Stream ~1.5× the documents
    # we need to be safe; we'll cut when accumulated tokens hit the target.
    approx_docs_needed = int(target_tokens / AVG_TOKENS["fineweb"] * 1.1)

    print(f"[fineweb] Loading stream "
          f"(target={target_tokens:,} tokens, ~{approx_docs_needed:,} docs) ...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        "sample-10BT",
        split="train",
        streaming=True,
    )
    count = 0
    accumulated_tokens = 0
    for row in ds:
        text = row.get("text", "")
        if not text:
            continue
        yield {"source": "fineweb", "text": text}
        # Rough token estimate (avg); we'll re-count exactly after tokenization
        accumulated_tokens += AVG_TOKENS["fineweb"]
        count += 1
        if accumulated_tokens >= target_tokens:
            break
        if count >= 2_000_000:  # absolute safety cap
            break
    print(f"[fineweb] Streamed {count:,} docs "
          f"(~{accumulated_tokens:,} estimated tokens).")


def stream_python_code_texts(max_rows: int) -> Iterator[Dict[str, str]]:
    """Stream Python source files from ajibawa-2023/Python-Code-Large.

    Applies a Python 3 filter: rows that appear to contain Python 2 syntax
    (bare ``print`` statements, old-style classes without ``from __future__``,
    ``u'...'`` unicode literals, etc.) are skipped.

    Yields {"source": "python_code", "text": <file content>}.
    """
    from datasets import load_dataset
    import re

    print(f"[python_code] Loading stream (cap={max_rows:,} docs) ...")
    ds = load_dataset(
        "ajibawa-2023/Python-Code-Large",
        "default",
        split="train",
        streaming=True,
    )

    # Python 2 detector — catches the most common patterns visible in the
    # dataset preview (python-twitter, web2py, simplejson, etc.).
    _PY2_PATTERNS = (
        # Bare `print "foo"` (no parentheses) — strongest signal
        re.compile(r'(?:^|\n)\s*print\s+["\']'),
        # Old-style class: `class Foo(object):` — works at start-of-string
        # or after a newline (the surrounding text in a file is preceded
        # by a newline in the vast majority of cases).
        re.compile(r'(?:^|\n)\s*class\s+\w+\s*\(\s*object\s*\)\s*:'),
        # `u'...'` unicode literal (Python 2 default str)
        re.compile(r'\bu["\']'),
        # `apply()` builtin removed in Python 3
        re.compile(r'\bapply\s*\('),
        # `exec` statement (not a function)
        re.compile(r"\bexec\s+['\"]"),
    )

    count = 0
    passed = 0
    for row in ds:
        code = row.get("code", "")
        if not code:
            continue

        # Python 3 filter: skip rows with Python 2 signals
        is_py2 = any(pat.search(code) for pat in _PY2_PATTERNS)
        if is_py2:
            count += 1
            continue

        yield {"source": "python_code", "text": code}
        passed += 1
        count += 1
        if count >= max_rows:
            break

    py2_pct = (count - passed) / max(count, 1) * 100
    print(f"[python_code] Streamed {passed:,} Python-3 docs "
          f"({py2_pct:.0f}% filtered as Python 2, {count:,} total examined).")


# ---------------------------------------------------------------------------
# Tokenization is straightforward — each source is large enough to fill its
# exact ratio; no redistribution needed.
# ---------------------------------------------------------------------------

@dataclass
class PlanOutput:
    """Final token targets per source after compute-optimal allocation."""
    ultradata_tokens:   int
    fineweb_tokens:   int
    python_code_tokens: int
    synthetic_docs:   int


def compute_adaptive_plan(
    cfg: TokenizeConfig,
) -> PlanOutput:
    """Decide final token targets (no adaptive redistribution needed).

    With three clean sources (swallow, fineweb, python_code) and no undersized
    datasets, the base 40/30/30 split is used directly.
    """
    print("\n[plan] Computing compute-optimal targets ...")

    target_ultradata  = int(TARGET_TOTAL_TOKENS * MATH_RATIO)       # 800M
    target_fineweb    = int(TARGET_TOTAL_TOKENS * GENERAL_RATIO)    # 600M
    target_python     = int(TARGET_TOTAL_TOKENS * PYTHON_CODE_RATIO) # 600M

    print(f"[plan] Targets: ultradata={target_ultradata:,}  "
          f"FineWeb={target_fineweb:,}  python_code={target_python:,}")

    return PlanOutput(
        ultradata_tokens=target_ultradata,
        fineweb_tokens=target_fineweb,
        python_code_tokens=target_python,
        synthetic_docs=cfg.max_synthetic,
    )


# ---------------------------------------------------------------------------
# Document cap helpers (pre-allocation)
# ---------------------------------------------------------------------------

def docs_for_tokens(target_tokens: int, source: str, safety: float = 1.15) -> int:
    """How many documents to fetch to hit target_tokens, with safety margin."""
    avg = AVG_TOKENS[source]
    return int(target_tokens / avg * safety)


# ---------------------------------------------------------------------------
# Tokenization wrapper
# ---------------------------------------------------------------------------

def build_tokenizer():
    """Build MathTokenizer."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.tokenizer.math_tokenizer import MathTokenizer
    return MathTokenizer()


def tokenize_iterator(
    text_iterator: Iterator[Dict[str, str]],
    tokenizer,
    targets: Dict[str, int],
    source_caps: Dict[str, int],
) -> Iterator[Dict]:
    """Tokenize text rows; cut each source at its target token count.

    targets    : token targets per source (used to cut streams)
    source_caps: hard document caps per source (safety bounds)
    """
    accumulated: Dict[str, int] = {s: 0 for s in targets}
    counts: Dict[str, int] = {s: 0 for s in targets}

    for record in text_iterator:
        source = record["source"]
        # Skip sources not in target plan (e.g. synthetic)
        if source not in targets:
            tokens = tokenizer.encode(record["text"], add_special=False)
            yield {
                "source":   source,
                "text":     record["text"],
                "tokens":   tokens,
                "n_tokens": len(tokens),
            }
            continue

        # Stop streaming this source if we've hit target
        if accumulated[source] >= targets[source]:
            continue
        # Hard cap safety
        if counts[source] >= source_caps[source]:
            continue

        tokens = tokenizer.encode(record["text"], add_special=False)
        n_tokens = len(tokens)
        accumulated[source] += n_tokens
        counts[source] += 1

        yield {
            "source":   source,
            "text":     record["text"],
            "tokens":   tokens,
            "n_tokens": n_tokens,
        }


# ---------------------------------------------------------------------------
# Parallel tokenization (workers > 1)
# ---------------------------------------------------------------------------

def _encode_batch(records: List[Dict]) -> List[Dict]:
    """Module-level worker: tokenize a small batch of records.

    Lives at module scope so multiprocessing can pickle it.  Each forked
    worker constructs its own MathTokenizer (tiktoken Encoding is
    fork-safe but not shareable across processes).
    """
    from src.tokenizer.math_tokenizer import MathTokenizer  # type: ignore
    tok = MathTokenizer()
    out = []
    for r in records:
        text = r["text"]
        try:
            ids = tok.encode(text, add_special=False)
        except Exception:
            # Skip records that fail to tokenize rather than crashing the worker
            continue
        out.append({
            "source":   r["source"],
            "text":     text,
            "tokens":   ids,
            "n_tokens": len(ids),
        })
    return out


def parallel_tokenize_iterator(
    text_iterator: Iterator[Dict],
    targets: Dict[str, int],
    source_caps: Dict[str, int],
    workers: int,
    batch_size: int = 32,
) -> Iterator[Dict]:
    """Producer/consumer tokenizer: source iter runs in main process,
    encoding runs in a ProcessPool.

    Cap tracking stays in the main process; workers tokenize small
    batches and the main process post-filters against target and
    document caps.  A small batch_size keeps wasted encode work
    bounded when a cap is hit mid-batch.
    """
    import multiprocessing as mp

    accumulated: Dict[str, int] = {src: 0 for src in targets}
    counts: Dict[str, int] = {src: 0 for src in source_caps}

    ctx = mp.get_context("fork")
    pool = ctx.Pool(processes=workers)

    def _batches():
        batch: List[Dict] = []
        for record in text_iterator:
            batch.append(record)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    try:
        n_yielded = 0
        t_start = time.time()
        for result_batch in pool.imap_unordered(
            _encode_batch, _batches(), chunksize=8
        ):
            for record in result_batch:
                source = record["source"]
                # Sources not in target plan (e.g. synthetic) bypass caps.
                if source not in targets:
                    yield record
                    n_yielded += 1
                    continue
                if accumulated[source] >= targets[source]:
                    continue
                if counts[source] >= source_caps[source]:
                    continue
                accumulated[source] += record["n_tokens"]
                counts[source] += 1
                yield record
                n_yielded += 1
                # Progress: log every 50k records and per-source milestones
                if n_yielded % 50_000 == 0:
                    elapsed = time.time() - t_start
                    rate = n_yielded / elapsed if elapsed > 0 else 0
                    src_summary = "  ".join(
                        f"{s}={accumulated.get(s,0)/1e6:.0f}M"
                        for s in sorted(accumulated)
                    )
                    print(f"  [tokenize] {n_yielded:,} records "
                          f"({rate:.0f} rec/s, {elapsed:.0f}s) | {src_summary}")
    except Exception as e:
        print(f"\n[tokenize] ERROR in pool: {e}")
        raise
    finally:
        pool.terminate()
        pool.join()


# ---------------------------------------------------------------------------
# Parquet chunking
# ---------------------------------------------------------------------------

def _estimate_row_bytes(n_tokens: int, source: str) -> int:
    """Rough byte estimate for one parquet row (tokens@2B + metadata + 30%)."""
    return int((n_tokens * 2 + len(source) + 24) * 1.3)


def write_parquet_chunks(
    token_iter: Iterator[Dict],
    output_dir: Path,
    rows_per_chunk: int,
    chunk_size_estimate: int,
) -> List[Path]:
    """Write tokenized rows to ~512 MB parquet chunks (zstd compressed)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_dir.mkdir(parents=True, exist_ok=True)

    sources: List[str] = []
    texts: List[str] = []
    tokens_list: List[List[int]] = []
    n_tokens_list: List[int] = []
    chunk_paths: List[Path] = []
    chunk_idx = 0
    current_estimated_bytes = 0
    t0 = time.time()

    # Running stats
    token_totals: Dict[str, int] = {}
    doc_totals:   Dict[str, int] = {}

    def _flush(idx: int) -> Path:
        nonlocal sources, texts, tokens_list, n_tokens_list
        nonlocal current_estimated_bytes, token_totals, doc_totals

        chunk_path = output_dir / f"chunk_{idx:04d}.parquet"
        table = pa.table({
            "source":   pa.array(sources,   type=pa.string()),
            "text":     pa.array(texts,     type=pa.string()),
            "tokens":   pa.array(tokens_list, type=pa.list_(pa.int32())),
            "n_tokens": pa.array(n_tokens_list, type=pa.int32()),
        })
        pq.write_table(table, chunk_path, compression="zstd")
        file_size = chunk_path.stat().st_size
        elapsed = time.time() - t0
        print(
            f"  chunk {idx:04d}: {len(sources):,} rows, "
            f"{file_size / 1024 / 1024:.1f} MB, elapsed={elapsed:.1f}s"
        )
        chunk_paths.append(chunk_path)
        sources = []
        texts = []
        tokens_list = []
        n_tokens_list = []
        current_estimated_bytes = 0
        return chunk_path

    for record in token_iter:
        sources.append(record["source"])
        texts.append(record["text"])
        tokens_list.append(record["tokens"])
        n_tokens_list.append(record["n_tokens"])
        current_estimated_bytes += _estimate_row_bytes(
            record["n_tokens"], record["source"]
        )

        token_totals[record["source"]] = (
            token_totals.get(record["source"], 0) + record["n_tokens"]
        )
        doc_totals[record["source"]] = doc_totals.get(record["source"], 0) + 1

        if (len(sources) >= rows_per_chunk
                or current_estimated_bytes >= chunk_size_estimate):
            _flush(chunk_idx)
            chunk_idx += 1

    if sources:
        _flush(chunk_idx)

    print(f"\n[parquet] Wrote {len(chunk_paths)} chunks:")
    for src, docs in sorted(doc_totals.items()):
        toks = token_totals.get(src, 0)
        print(f"  {src:<10}: {docs:>7,} docs, {toks:>12,} tokens "
              f"({toks / 1e6:.1f}M)")
    return chunk_paths


# ---------------------------------------------------------------------------
# Local-disk prefetch (decouples HF network throughput from worker count)
# ---------------------------------------------------------------------------

def _prefetch_to_disk(
    cfg: "PretokenizeConfig",
    plan: "Plan",
    cache_dir: Path,
) -> Dict[str, Path]:
    """Download all HF datasets to local JSONL shards BEFORE tokenization.

    Why: with 24 workers, each forked worker is sharing the main process's
    HTTP connection pool to HF's CDN.  HF throttles per-IP concurrent
    downloads (typically 8-16), so beyond that you get slowdown AND
    "peer closed connection" errors.  Tokenization is CPU-bound, not
    network-bound, so we should keep workers focused on tokenizing and
    pull the network bytes in serial.

    After this returns, the tokenization phase reads from local disk
    (essentially free I/O), so all 24 workers stay busy on the bottleneck.

    Idempotent: re-running the script skips already-complete shards.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    caps = {
        "ultradata_math": cfg.max_swallow,
        "python_code":    cfg.max_python_code,
        "fineweb":        docs_for_tokens(plan.fineweb_tokens, "fineweb"),
    }

    sources: List[Tuple[str, int]] = [
        ("synthetic",      cfg.max_synthetic),
        ("ultradata_math", cfg.max_swallow),
        ("python_code",    cfg.max_python_code),
        ("fineweb",        caps["fineweb"]),
    ]

    iterators = {
        "synthetic":      lambda n: stream_synthetic_texts(n, seed=42),
        "ultradata_math": lambda n: stream_ultradata_texts(n),
        "python_code":    lambda n: stream_python_code_texts(n),
        "fineweb":        lambda n: stream_fineweb_texts(plan.fineweb_tokens),
    }

    paths: Dict[str, Path] = {}
    for source, n in sources:
        shard_path = cache_dir / f"{source}.jsonl"
        # Idempotency: skip if a complete shard already exists.
        if shard_path.exists() and shard_path.stat().st_size > 0:
            existing = sum(1 for _ in shard_path.open("rb"))
            print(f"[prefetch] {source}: reusing {shard_path} ({existing:,} rows)")
            paths[source] = shard_path
            continue

        print(f"[prefetch] {source}: downloading to {shard_path} ...")
        t0 = time.time()
        count = 0
        bytes_written = 0
        with shard_path.open("w", encoding="utf-8") as f:
            for record in iterators[source](n):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                bytes_written += len(record.get("text", "")) + 32
                count += 1
                if count % 5000 == 0:
                    elapsed = time.time() - t0
                    rate = count / elapsed if elapsed > 0 else 0
                    print(f"[prefetch] {source}: {count:,} rows "
                          f"({rate:.0f} rows/s, {bytes_written / 1e6:.1f} MB)")
        elapsed = time.time() - t0
        print(f"[prefetch] {source}: wrote {count:,} rows in "
              f"{elapsed:.1f}s ({bytes_written / 1e6:.1f} MB)")
        paths[source] = shard_path

    return paths


def _disk_iter(path: Path) -> Iterator[Dict[str, str]]:
    """Replay a JSONL shard back as the original text iterator."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ---------------------------------------------------------------------------
# HF Hub upload
# ---------------------------------------------------------------------------

def _get_hf_token(token_env: str) -> Optional[str]:
    """Read HF token from environment or .env file.

    Returns the token string, or None if not configured.  This is
    NEVER raised as an error — tokenization does not require a token;
    only the optional HF Hub upload step does.
    """
    token = os.environ.get(token_env)
    if token:
        return token
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith(f"{token_env}="):
                token = line.split("=", 1)[1].strip().strip('"').strip("'")
                if token:
                    return token
    return None


def upload_to_hub(
    chunk_paths: List[Path],
    repo_id: str,
    token: str,
    plan: PlanOutput,
) -> None:
    """Upload chunks + manifest to HF Hub."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=repo_id, repo_type=HF_REPO_TYPE, exist_ok=True)
    except Exception as exc:
        raise RuntimeError(f"Cannot access/create repo '{repo_id}': {exc}") from exc

    print(f"[hf] Uploading {len(chunk_paths)} chunks to {repo_id}/data/train/")
    for i, chunk_path in enumerate(chunk_paths):
        remote_path = f"data/train/{chunk_path.name}"
        print(f"  [{i+1}/{len(chunk_paths)}] → {chunk_path.name}")
        api.upload_file(
            path_or_fileobj=str(chunk_path),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type=HF_REPO_TYPE,
        )

    manifest = {
        "repo_id":       repo_id,
        "total_chunks":  len(chunk_paths),
        "target_total_tokens": TARGET_TOTAL_TOKENS,
        "split_ratios":  {
            "ultradata_math": MATH_RATIO,
            "fineweb":        GENERAL_RATIO,
            "python_code":    PYTHON_CODE_RATIO,
        },
        "plan": {
            "ultradata_tokens":    plan.ultradata_tokens,
            "fineweb_tokens":     plan.fineweb_tokens,
            "python_code_tokens": plan.python_code_tokens,
            "synthetic_docs":    plan.synthetic_docs,
        },
        "avg_tokens_per_doc": AVG_TOKENS,
        "sources": {
            "ultradata_math": "issdandavis/UltraData-Math (L2-preview, quality>=3, 4 shards ≈ 400K docs)",
            "fineweb":     "HuggingFaceFW/fineweb (sample-10BT)",
            "python_code": "ajibawa-2023/Python-Code-Large (Python 3 filtered, max 500k docs)",
            "synthetic":   "SyntheticMathDataset (seed=42, max_digits=3)",
        },
        "benchmarks": {
            "note": "GSM8K and MATH (hendrycks/competition_math) are EVALUATION benchmarks only. "
                    "They must NEVER be used in any training data.",
        },
    }
    manifest_path = LOCAL_DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    api.upload_file(
        path_or_fileobj=str(manifest_path),
        path_in_repo="data/train/manifest.json",
        repo_id=repo_id,
        repo_type=HF_REPO_TYPE,
    )
    print(f"\n[huggingface] Upload complete → "
          f"https://huggingface.co/datasets/{repo_id}")


def _count_rows_pq(path: Path) -> int:
    import pyarrow.parquet as pq
    return pq.ParquetFile(str(path)).metadata.num_rows


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_tokenize(cfg: TokenizeConfig) -> None:
    """Full pretokenization pipeline."""
    t_start = time.time()
    print("=" * 70)
    print("RWKV-7 math pretokenization pipeline")
    print("=" * 70)
    print(f"Output dir       : {cfg.output_dir}")
    print(f"HF repo          : {HF_REPO_ID}")
    print(f"Target total     : {TARGET_TOTAL_TOKENS / 1e9:.2f}B tokens")
    print(f"Split (target)   : math={MATH_RATIO*100:.0f}% "
          f"general={GENERAL_RATIO*100:.0f}% python_code={PYTHON_CODE_RATIO*100:.0f}%")
    print()

    # 1. Resolve HF token (optional — only required if you want to upload)
    hf_token = None if cfg.no_upload else _get_hf_token(cfg.hf_token_env)
    if cfg.no_upload:
        print("[hf] --no-upload set — upload step will be SKIPPED.")
        print("[hf] Parquet chunks will be kept locally at:", cfg.output_dir, "\n")
    elif hf_token:
        print(f"[hf] Token loaded from {cfg.hf_token_env} — upload will run.\n")
    else:
        print("[hf] No HF_TOKEN set — upload will be SKIPPED.")
        print("[hf] Parquet chunks will be kept locally at:", cfg.output_dir)
        print("[hf] Set HF_TOKEN in your shell and re-run to upload.\n")

    # 2. Build tokenizer (used by sequential path; parallel workers build
    #    their own copies inside the forked pool)
    print("[tokenizer] Loading MathTokenizer ...")
    tokenizer = build_tokenizer()
    print(f"[tokenizer] vocab={tokenizer.n_vocab}\n")

    # 3. Compute plan (no adaptive redistribution needed with clean sources)
    plan = compute_adaptive_plan(cfg)

    # 4. Build source iterators + target dict + cap dict
    targets = {
        "ultradata_math": plan.ultradata_tokens,
        "fineweb":        plan.fineweb_tokens,
        "python_code":    plan.python_code_tokens,
    }
    caps = {
        "ultradata_math": cfg.max_swallow,
        "python_code":    cfg.max_python_code,
        "fineweb":        docs_for_tokens(plan.fineweb_tokens, "fineweb"),
    }
    print(f"\n[plan] Document caps: {caps}")
    print(f"[plan] Workers: {cfg.workers} "
          f"(batch_size={cfg.encode_batch_size})\n")

    # 5. Prefetch HF datasets to local JSONL shards (single-threaded;
    #    HF's CDN throttles concurrent downloads, so serial is faster
    #    than 24-way parallel and avoids "peer closed connection" drops).
    cache_dir = cfg.output_dir.parent / "prefetch_cache"
    shards = _prefetch_to_disk(cfg, plan, cache_dir)
    print()

    # 6. Build source iterator (synthetic first, no cuts)
    def all_sources():
        # Synthetic first — fully local, always available
        yield from _disk_iter(shards["synthetic"])
        # UltraData-Math (quality-filtered forum math)
        yield from _disk_iter(shards["ultradata_math"])
        # Python code (already Python-3 filtered)
        yield from _disk_iter(shards["python_code"])
        # FineWeb — already cut to target
        yield from _disk_iter(shards["fineweb"])

    # 7. Tokenize + write parquet
    print("[tokenize] Starting tokenization (reads from local prefetch cache) ...")
    if cfg.workers > 1:
        print(f"[tokenize] Parallel mode: {cfg.workers} workers, "
              f"batch={cfg.encode_batch_size}")
        token_iter = parallel_tokenize_iterator(
            all_sources(), targets, caps,
            workers=cfg.workers,
            batch_size=cfg.encode_batch_size,
        )
    else:
        print("[tokenize] Sequential mode (workers=1)")
        token_iter = tokenize_iterator(all_sources(), tokenizer, targets, caps)
    chunk_paths = write_parquet_chunks(
        token_iter, cfg.output_dir,
        cfg.rows_per_chunk, cfg.chunk_size_estimate,
    )

    # 7. Upload (only if HF token is available and not disabled)
    if cfg.no_upload:
        print("\n[huggingface] Skipped upload (--no-upload).")
        print(f"[huggingface] Parquet chunks kept locally in: {cfg.output_dir}")
        print(f"[huggingface] {len(chunk_paths)} chunk(s) ready for upload later.")
    elif hf_token:
        print("\n[huggingface] Uploading ...")
        upload_to_hub(chunk_paths, HF_REPO_ID, hf_token, plan)

        # 8. Cleanup local parquet (only after successful upload)
        for p in chunk_paths:
            p.unlink()
        manifest_local = LOCAL_DATA_DIR / "manifest.json"
        if manifest_local.exists():
            manifest_local.unlink()
        try:
            LOCAL_DATA_DIR.rmdir()
        except OSError:
            pass
    else:
        print("\n[huggingface] Skipped upload (no HF_TOKEN).")
        print(f"[huggingface] Parquet chunks kept locally in: {cfg.output_dir}")
        print(f"[huggingface] {len(chunk_paths)} chunk(s) ready for upload later.")

    elapsed = time.time() - t_start
    print(f"\n=== Done in {elapsed:.1f}s ===")


# ---------------------------------------------------------------------------
# Modal deployment
# ---------------------------------------------------------------------------

if is_modal:
    modal_app = modal.App("rwkv7-pretok")

    PRETOK_IMAGE = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch>=2.1.0",
            "einops>=0.7.0",
            "numpy>=1.24",
            "tiktoken>=0.5.0",
            "datasets>=2.14.0",
            "huggingface-hub>=0.20.0",
            "safetensors>=0.4.0",
            "pyarrow>=14.0.0",
            "zstandard>=0.21.0",
        )
    )

    @modal_app.function(
        image=PRETOK_IMAGE,
        cpu=32,
        memory=32768,
        timeout=3600 * 12,
        retries=modal.Retries(max_retries=2),
    )
    def upload_pretokenized(
        max_swallow: int = 50_000,
        max_python_code: int = 500_000,
        max_synthetic: int = 2_000,
    ):
        """Run pretokenization on a 32-CPU Modal VM and upload to HF Hub."""
        cfg = TokenizeConfig(
            max_swallow=max_swallow,
            max_python_code=max_python_code,
            max_synthetic=max_synthetic,
            output_dir=Path("/tmp/parquet_chunks"),
        )
        run_tokenize(cfg)


# ---------------------------------------------------------------------------
# Local entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Pretokenize the 2B-token pretraining corpus. "
                    "By default uploads to HF Hub if HF_TOKEN is set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--max-swallow",     type=int, default=50_000)
    parser.add_argument("--max-python-code", type=int, default=500_000)
    parser.add_argument("--max-synthetic",   type=int, default=2_000)
    parser.add_argument(
        "--workers", type=int,
        default=os.cpu_count() or 1,
        help="Number of encoding workers (processes). Default: os.cpu_count().",
    )
    parser.add_argument(
        "--encode-batch-size", type=int, default=32,
        help="Records per pool task — smaller = lower wasted work when a "
             "cap is hit mid-batch.",
    )
    parser.add_argument(
        "--no-upload", action="store_true",
        help="Skip the HF Hub upload even if HF_TOKEN is set. Chunks are "
             "kept locally in ./parquet_chunks/.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=LOCAL_DATA_DIR,
        help="Where to write parquet chunks before upload.",
    )
    parser.add_argument("--local", action="store_true",
                        help="Force the local path even if modal is installed.")
    args = parser.parse_args()

    cfg = TokenizeConfig(
        max_swallow=args.max_swallow,
        max_python_code=args.max_python_code,
        max_synthetic=args.max_synthetic,
        workers=args.workers,
        encode_batch_size=args.encode_batch_size,
        no_upload=args.no_upload,
        output_dir=args.output_dir,
    )

    if args.local or not is_modal:
        run_tokenize(cfg)
    else:
        upload_pretokenized.remote(
            max_swallow=cfg.max_swallow,
            max_synthetic=cfg.max_synthetic,
        )
