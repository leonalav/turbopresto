"""Pretokenization pipeline for RWKV-7 math model (50M parameters).

Fetches data from:
  - open-r1/OpenR1-Math-220k  (94k math problems with R1 reasoning traces)
  - camel-ai/physics           (physics dialogues — small dataset, fully used)
  - HuggingFaceFW/fineweb (sample/10BT) — general English web text

────────────────────────────────────────────────────────────────────────────
Compute-optimal sizing (per Chinchilla / Hoffmann et al. 2022):
  Target tokens  = 40 × N_params ≈ 40 × 50M = 2.0 B tokens
  (Chinchilla is 20×; we use 40× because math/scientific corpora are denser
   and we want extra reasoning signal.)
────────────────────────────────────────────────────────────────────────────

Target split: 40% math / 30% general / 30% physics
  - OpenR1-Math-220k  → 40% × 2.0B = 800M tokens
  - FineWeb-10BT       → 30% × 2.0B = 600M tokens
  - camel-ai/physics   → 30% × 2.0B = 600M tokens target

BUT camel-ai/physics only contains ~1.5M tokens (23.5 MB compressed).
So we use it ALL, then redistribute the surplus:
  physics_tokens_actual  = 1.5M   (use everything)
  surplus                = 600M − 1.5M = 598.5M
  redistribute to OpenR1 : FineWeb in 40 : 30 ratio
    OpenR1_extra   = 598.5M × (40/70) ≈ 342M
    FineWeb_extra  = 598.5M × (30/70) ≈ 257M
  Final targets:
    OpenR1  : 800M + 342M = 1.142B tokens
    FineWeb : 600M + 257M = 857M tokens
    physics : 1.5M tokens  (full dataset)

────────────────────────────────────────────────────────────────────────────

Tokenizes with MathTokenizer and saves as Parquet files (~512 MB each).
Uploads to:
  https://huggingface.co/datasets/leonidas123/valkmodel-data

Usage:
  # Local (requires .env with HF_TOKEN)
  python pretok/tokenize.py

  # Modal 32-CPU VM
  modal run pretok/tokenize.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

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
MATH_RATIO          = 0.40                 # OpenR1
GENERAL_RATIO       = 0.30                 # FineWeb
PHYSICS_RATIO       = 0.30                 # camel-ai/physics

# ── Avg tokens per document (used for pre-allocation to avoid buffering) ───
AVG_TOKENS = {
    "openr1":    25_000,   # long R1 traces
    "fineweb":     700,    # typical web paragraph
    "physics":     700,    # one Q+A turn
    "synthetic":    20,    # arithmetic sentence
}


@dataclass
class TokenizeConfig:
    """Configuration for the pretokenization run."""
    # Source caps (in DOCUMENTS, hard upper bound)
    max_openr1:    int = 50_000
    max_physics:   int = 30_000
    max_synthetic: int = 2_000

    hf_token_env: str = "HF_TOKEN"
    output_dir:   Path = field(default_factory=lambda: LOCAL_DATA_DIR)
    rows_per_chunk: int = ROWS_PER_CHUNK
    chunk_size_estimate: int = CHUNK_SIZE_ESTIMATE_BYTES


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


def stream_openr1_texts(max_rows: int) -> Iterator[Dict[str, str]]:
    """Stream math problems from open-r1/OpenR1-Math-220k.

    Raises RuntimeError on failure (caller must handle).
    Yields {"source": "openr1", "text": <problem + solution + answer>}.
    """
    from datasets import load_dataset
    print(f"[openr1] Loading stream (cap={max_rows:,} docs) ...")
    ds = load_dataset(
        "open-r1/OpenR1-Math-220k",
        "default",
        split="train",
        streaming=True,
    )
    count = 0
    for row in ds:
        problem = row.get("problem", "")
        solution = row.get("solution", "")
        answer = row.get("answer", "")
        if not problem or not solution:
            continue
        text = f"Problem: {problem}\n\nSolution:\n{solution}\n\nAnswer: {answer}"
        yield {"source": "openr1", "text": text}
        count += 1
        if count >= max_rows:
            break
    print(f"[openr1] Streamed {count:,} rows.")


def stream_physics_texts(max_rows: int) -> Iterator[Dict[str, str]]:
    """Stream ALL physics dialogues from camel-ai/physics.

    Note: this dataset is small (23.5 MB, ~1.5M tokens).  We typically use
    the full corpus; `max_rows` is only a safety bound.

    Yields {"source": "physics", "text": <user + assistant>}.
    """
    from datasets import load_dataset
    print(f"[physics] Loading stream (cap={max_rows:,} docs) ...")
    ds = load_dataset("camel-ai/physics", split="train", streaming=True)

    pending_user: Optional[str] = None
    pending_field: Optional[str] = None
    count = 0
    total_rows = 0

    for row in ds:
        total_rows += 1
        role = str(row.get("role_type", "")).upper()
        msg = row.get("message", {})
        content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
        if not content:
            continue
        physics_field = str(row.get("physics_field", "unknown"))

        if "USER" in role:
            pending_user = f"[Physics: {physics_field}] {content}"
            pending_field = physics_field
        elif "ASSISTANT" in role and pending_user is not None:
            text = f"{pending_user}\n\nAssistant:\n{content}"
            yield {"source": "physics", "text": text}
            pending_user = None
            count += 1
            if count >= max_rows:
                break

    print(f"[physics] Streamed {count:,} dialogues from {total_rows:,} raw rows.")


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


# ---------------------------------------------------------------------------
# Adaptive sampling: physics is small → use all, redistribute the rest
# ---------------------------------------------------------------------------

@dataclass
class PlanOutput:
    """Final token targets per source after adaptive redistribution."""
    openr1_tokens:   int
    fineweb_tokens:  int
    physics_tokens:  int    # target (will be capped at actual)
    physics_actual:  int    # real count after streaming
    synthetic_docs:   int


def compute_adaptive_plan(cfg: TokenizeConfig) -> PlanOutput:
    """Decide final token targets based on actual physics dataset size.

    Strategy:
      1. Try to estimate physics size cheaply (by streaming first N rows).
      2. Compute surplus: physics_target − physics_actual.
      3. Redistribute surplus to OpenR1 and FineWeb in 40:30 ratio.
      4. Final tokens per source → convert to docs via AVG_TOKENS.
    """
    print("\n[plan] Computing compute-optimal targets ...")

    # 1. Target tokens
    target_math     = int(TARGET_TOTAL_TOKENS * MATH_RATIO)        # 800M
    target_general  = int(TARGET_TOTAL_TOKENS * GENERAL_RATIO)     # 600M
    target_physics  = int(TARGET_TOTAL_TOKENS * PHYSICS_RATIO)     # 600M
    print(f"[plan] Base targets: OpenR1={target_math:,}  "
          f"FineWeb={target_general:,}  physics={target_physics:,}")

    # 2. Estimate physics size by sampling a bounded prefix via .take(N).
    #    This is a single pass over a small slice of the corpus instead of
    #    a full re-stream just to count rows.
    print("[plan] Estimating actual physics dataset size ...")
    from datasets import load_dataset
    physics_ds = load_dataset("camel-ai/physics", split="train", streaming=True)
    # .take() returns a generator; consume it once into a list.
    # cfg.max_physics is the same safety bound used in all_sources().
    physics_sample = list(physics_ds.take(cfg.max_physics))
    physics_count = len(physics_sample)
    physics_actual_tokens = physics_count * AVG_TOKENS["physics"]
    sample_texts = [r.get("message", {}).get("content", "")[:80]
                    for r in physics_sample[:10]
                    if r.get("message")]

    print(f"[plan] Physics: {physics_count:,} dialogues (capped at {cfg.max_physics:,}), "
          f"~{physics_actual_tokens:,} estimated tokens")
    if sample_texts:
        print(f"[plan] Sample physics texts:")
        for s in sample_texts[:3]:
            print(f"         {s!r}")

    # 3. Compute surplus and redistribute
    if physics_actual_tokens < target_physics:
        surplus = target_physics - physics_actual_tokens
        # redistribute 40:30 between OpenR1 and FineWeb
        openr1_extra  = int(surplus * (MATH_RATIO /
                                       (MATH_RATIO + GENERAL_RATIO)))
        fineweb_extra = surplus - openr1_extra
        final_math    = target_math + openr1_extra
        final_general = target_general + fineweb_extra
        final_physics = physics_actual_tokens
        print(f"[plan] Physics undersized. Surplus={surplus:,}")
        print(f"[plan]   → OpenR1 : {target_math:,} + {openr1_extra:,} "
              f"= {final_math:,}")
        print(f"[plan]   → FineWeb: {target_general:,} + {fineweb_extra:,} "
              f"= {final_general:,}")
        print(f"[plan]   → physics: {final_physics:,} (full)")
    else:
        final_math    = target_math
        final_general = target_general
        final_physics = target_physics
        print(f"[plan] Physics meets target. No redistribution.")

    return PlanOutput(
        openr1_tokens=final_math,
        fineweb_tokens=final_general,
        physics_tokens=final_physics,
        physics_actual=physics_actual_tokens,
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
# HF Hub upload
# ---------------------------------------------------------------------------

def _get_hf_token(token_env: str) -> str:
    """Read HF token from environment or .env file."""
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
    raise RuntimeError(
        f"{token_env} not set in environment or {env_path}. "
        f"Cannot upload to HF Hub."
    )


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
        "split_ratios":  {"openr1": MATH_RATIO, "fineweb": GENERAL_RATIO,
                          "physics": PHYSICS_RATIO},
        "plan": {
            "openr1_tokens":  plan.openr1_tokens,
            "fineweb_tokens": plan.fineweb_tokens,
            "physics_tokens": plan.physics_tokens,
            "physics_actual": plan.physics_actual,
            "synthetic_docs": plan.synthetic_docs,
        },
        "avg_tokens_per_doc": AVG_TOKENS,
        "sources": {
            "openr1":    "open-r1/OpenR1-Math-220k (default split)",
            "physics":   "camel-ai/physics (full dataset)",
            "fineweb":   "HuggingFaceFW/fineweb (sample-10BT)",
            "synthetic": "SyntheticMathDataset (seed=42, max_digits=3)",
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
          f"general={GENERAL_RATIO*100:.0f}% physics={PHYSICS_RATIO*100:.0f}%")
    print()

    # 1. Validate HF token
    hf_token = _get_hf_token(cfg.hf_token_env)
    print("[hf] Token loaded.\n")

    # 2. Build tokenizer
    print("[tokenizer] Loading MathTokenizer ...")
    tokenizer = build_tokenizer()
    print(f"[tokenizer] vocab={tokenizer.n_vocab}\n")

    # 3. Compute adaptive plan
    plan = compute_adaptive_plan(cfg)

    # 4. Build source iterators + target dict + cap dict
    targets = {
        "openr1":   plan.openr1_tokens,
        "fineweb":  plan.fineweb_tokens,
        "physics":  plan.physics_tokens,
    }
    caps = {
        "openr1":   cfg.max_openr1,
        "physics":  cfg.max_physics,    # safety
        "fineweb":  docs_for_tokens(plan.fineweb_tokens, "fineweb"),
    }
    print(f"\n[plan] Document caps: {caps}")

    # 5. Build full source iterator (synthetic first, no cuts)
    print("\n[stream] Reading sources ...")

    def all_sources():
        # Synthetic always included, no token cap (only 2000 docs total)
        yield from stream_synthetic_texts(cfg.max_synthetic, seed=42)
        # Then physics, OpenR1, FineWeb (in priority order; tokens are tracked)
        # Physics is fully streamed (plan.physics_actual)
        yield from stream_physics_texts(cfg.max_physics)
        # OpenR1 — up to cfg.max_openr1 docs
        yield from stream_openr1_texts(cfg.max_openr1)
        # FineWeb — streams until target_tokens hit
        yield from stream_fineweb_texts(plan.fineweb_tokens)

    # 6. Tokenize + write parquet
    print("\n[tokenize] Starting tokenization ...")
    token_iter = tokenize_iterator(all_sources(), tokenizer, targets, caps)
    chunk_paths = write_parquet_chunks(
        token_iter, cfg.output_dir,
        cfg.rows_per_chunk, cfg.chunk_size_estimate,
    )

    # 7. Upload
    print("\n[huggingface] Uploading ...")
    upload_to_hub(chunk_paths, HF_REPO_ID, hf_token, plan)

    # 8. Cleanup local parquet
    for p in chunk_paths:
        p.unlink()
    manifest_local = LOCAL_DATA_DIR / "manifest.json"
    if manifest_local.exists():
        manifest_local.unlink()
    try:
        LOCAL_DATA_DIR.rmdir()
    except OSError:
        pass

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
        max_openr1: int = 50_000,
        max_physics: int = 30_000,
        max_synthetic: int = 2_000,
    ):
        """Run pretokenization on a 32-CPU Modal VM and upload to HF Hub."""
        cfg = TokenizeConfig(
            max_openr1=max_openr1,
            max_physics=max_physics,
            max_synthetic=max_synthetic,
            output_dir=Path("/tmp/parquet_chunks"),
        )
        run_tokenize(cfg)


# ---------------------------------------------------------------------------
# Local entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-openr1", type=int, default=50_000)
    parser.add_argument("--max-physics", type=int, default=30_000)
    parser.add_argument("--max-synthetic", type=int, default=2_000)
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()

    cfg = TokenizeConfig(
        max_openr1=args.max_openr1,
        max_physics=args.max_physics,
        max_synthetic=args.max_synthetic,
    )

    if args.local or not is_modal:
        run_tokenize(cfg)
    else:
        upload_pretokenized.remote(
            max_openr1=cfg.max_openr1,
            max_physics=cfg.max_physics,
            max_synthetic=cfg.max_synthetic,
        )
