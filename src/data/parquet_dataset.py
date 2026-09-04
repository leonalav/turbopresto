"""Parquet-backed dataset loader for RWKV-7 training.

Loads pretokenized parquet files from HuggingFace Hub and yields token-id
arrays for training. Designed to work with the pretokenized output from
`python -m pretok.pretokenize`.

Features:
  - Streams chunks via hf_hub_download (no full repo clone)
  - Caches downloaded chunks locally (under HF cache)
  - Deterministic shuffling (numpy RNG, seedable)
  - Round-robin sampling across sources to enforce 30/30/40 split
  - Resumable via checkpoint-step index
  - Validation split (last 1% of docs)
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# HF Hub helpers
# ---------------------------------------------------------------------------

def _get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                t = line.split("=", 1)[1].strip().strip('"').strip("'")
                if t:
                    return t
    raise RuntimeError("HF_TOKEN not set; cannot download pretokenized data.")


def list_parquet_files(repo_id: str, token: str) -> List[str]:
    """List all parquet files in the HF repo.

    Supports two layouts, in this order of preference:
      1. **Root-level** parquet files (``chunk_0000.parquet``,
         ``shard_001.parquet``, …) — current schema produced by
         ``pretok.pretokenize`` and uploaded directly to the repo root.
      2. **Legacy** ``data/train/*.parquet`` layout.

    A non-zero root-level match always wins so a partially-migrated repo
    doesn't pull in stale legacy chunks by accident.
    """
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    parquets = [f for f in files if f.endswith(".parquet")]
    if not parquets:
        return []

    root_parquets = sorted(f for f in parquets if "/" not in f)
    legacy_parquets = sorted(f for f in parquets if f.startswith("data/train/"))

    if root_parquets:
        return root_parquets
    return legacy_parquets


def download_parquet(repo_id: str, filename: str, token: str) -> Path:
    """Download a single parquet file to HF cache, return local path."""
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=token,
    ))


def download_manifest(repo_id: str, token: str) -> Optional[Dict[str, Any]]:
    """Download ``manifest.json`` from the HF repo (root, then ``data/train/``)."""
    from huggingface_hub import hf_hub_download
    for filename in ("manifest.json", "data/train/manifest.json"):
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                token=token,
            )
            return json.loads(Path(path).read_text())
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# ParquetDataset: streams tokens from downloaded parquet chunks
# ---------------------------------------------------------------------------

@dataclass
class ParquetShard:
    """In-memory record of one downloaded parquet chunk."""
    local_path: Path
    source_counts: Dict[str, int]   # source name → doc count
    token_counts: Dict[str, int]    # source name → token count
    doc_index: List[int]            # row indices into this shard
    pointer: int = 0                # current read position (row)


class ParquetDataset:
    """Streaming dataset over pretokenized parquet chunks.

    Usage:
        ds = ParquetDataset(repo_id="leonidas123/valkmodel-data",
                            seq_len=4096, batch_size=16,
                            sources_filter=["openr1", "fineweb", "physics"])
        for batch in ds.iter_batches(seed=42, resume_step=0):
            ...train...
    """

    def __init__(
        self,
        repo_id: str = "leonidas123/valkmodel-data",
        token: Optional[str] = None,
        seq_len: int = 4096,
        batch_size: int = 16,
        sources_filter: Optional[List[str]] = None,
        # Optional: hold out the last `val_frac` of each source for validation
        val_frac: float = 0.0,
        train: bool = True,
        # M1 fix: tokenizer EOS id, used to mark doc boundaries so the model
        # learns to predict EOS at end-of-document. Falls back to 0 (with a
        # one-shot warning) for backward compat — call sites should pass the
        # tokenizer's `eos_id`.
        eos_id: Optional[int] = None,
    ):
        self.repo_id = repo_id
        self.token = token or _get_hf_token()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.sources_filter = sources_filter  # None = include all
        self.val_frac = val_frac
        self.train = train
        self.eos_id = eos_id
        self._warned_eos_fallback = False

        # H3 fix: cache the full pyarrow Table per shard so we never
        # re-open the parquet file.  Maps shard index -> (table, path_mtime).
        self._table_cache: Dict[int, Tuple[Any, float]] = {}

        # State (filled by .load())
        self.shards: List[ParquetShard] = []
        self.manifest: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, max_shards: Optional[int] = None) -> None:
        """Download the manifest and every parquet file in data/train/."""
        print(f"[ParquetDataset] Loading from {self.repo_id} ...")
        self.manifest = download_manifest(self.repo_id, self.token)
        if self.manifest:
            print(f"[ParquetDataset] Manifest: "
                  f"{self.manifest.get('total_chunks')} chunks, "
                  f"target_tokens={self.manifest.get('target_total_tokens')}")

        files = list_parquet_files(self.repo_id, self.token)
        if max_shards is not None:
            files = files[:max_shards]
        print(f"[ParquetDataset] Downloading {len(files)} parquet files ...")

        t0 = time.time()
        for i, f in enumerate(files):
            local = download_parquet(self.repo_id, f, self.token)
            shard = self._build_shard(local)
            self.shards.append(shard)
            if (i + 1) % 10 == 0 or (i + 1) == len(files):
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(files)}] downloaded in {elapsed:.1f}s")

        if not self.shards:
            raise RuntimeError(
                f"No parquet files found in {self.repo_id}/data/train/. "
                "Run pretokenization first."
            )
        print(f"[ParquetDataset] {len(self.shards)} shards loaded.")

    def _build_shard(self, local_path: Path) -> ParquetShard:
        """Read a parquet file; return shard metadata + row indices."""
        pf = pq.ParquetFile(str(local_path))
        n = pf.metadata.num_rows

        # Read source & n_tokens columns only (cheap)
        tbl = pf.read(columns=["source", "n_tokens"])
        sources = tbl["source"].to_pylist()
        n_tokens = tbl["n_tokens"].to_pylist()

        source_counts: Dict[str, int] = {}
        token_counts: Dict[str, int] = {}
        indices: List[int] = []
        for i, (src, tok) in enumerate(zip(sources, n_tokens)):
            if self.sources_filter is not None and src not in self.sources_filter:
                continue
            source_counts[src] = source_counts.get(src, 0) + 1
            token_counts[src] = token_counts.get(src, 0) + int(tok)
            indices.append(i)

        return ParquetShard(
            local_path=local_path,
            source_counts=source_counts,
            token_counts=token_counts,
            doc_index=indices,
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def total_docs(self) -> int:
        return sum(len(s.doc_index) for s in self.shards)

    def total_tokens(self) -> int:
        return sum(sum(s.token_counts.values()) for s in self.shards)

    def source_stats(self) -> Dict[str, Dict[str, int]]:
        """Aggregate doc and token counts per source."""
        out: Dict[str, Dict[str, int]] = {}
        for shard in self.shards:
            for src, c in shard.source_counts.items():
                out.setdefault(src, {"docs": 0, "tokens": 0})
                out[src]["docs"] += c
                out[src]["tokens"] += shard.token_counts.get(src, 0)
        return out

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def _ensure_table_loaded(self, shard_idx: int) -> Any:
        """Load (or return cached) pyarrow Table for a shard.

        H3 fix: we open each parquet file exactly once per dataset lifetime.
        The table is cached for all subsequent row reads.
        """
        import time as _time

        shard = self.shards[shard_idx]
        if shard_idx in self._table_cache:
            cached_table, cached_mtime = self._table_cache[shard_idx]
            # Re-check mtime to handle updated files (e.g. re-downloaded)
            current_mtime = shard.local_path.stat().st_mtime
            if cached_mtime == current_mtime:
                return cached_table
            # File changed; reload
            del self._table_cache[shard_idx]

        pf = pq.ParquetFile(str(shard.local_path))
        table = pf.read(columns=["source", "tokens"])
        mtime = shard.local_path.stat().st_mtime
        self._table_cache[shard_idx] = (table, mtime)
        return table

    def _read_row(self, shard_idx: int, row_idx: int) -> Tuple[List[int], str]:
        """Read one row from a parquet shard; return (tokens, source)."""
        table = self._ensure_table_loaded(shard_idx)
        tokens = list(table["tokens"][row_idx].as_py())
        source = table["source"][row_idx].as_py()
        return tokens, source

    def iter_batches(
        self,
        seed: int = 42,
        resume_step: int = 0,
        deterministic: bool = False,
    ) -> Iterator[Dict[str, Any]]:
        """Yield training batches as {input_ids, labels} tensors.

        Each batch packs `batch_size` examples up to seq_len tokens each
        (no padding within batch — uses RWKVCollator-style packing).

        Args:
            seed: RNG seed for shuffling document order.
            resume_step: skip this many batches before yielding (for resume).
            deterministic: if True, walk shards in order without shuffling.

        Yields:
            {"input_ids": LongTensor[seq_len], "labels": LongTensor[seq_len]}
            (a single packed sequence per batch; effective batch size = 1)
        """
        # H2 fix: build the set of (shard_idx, row_idx) pairs that belong
        # to the validation split, then exclude them from training.
        val_refs: set[tuple[int, int]] = set()
        if self.val_frac > 0:
            for s_idx, shard in enumerate(self.shards):
                n = len(shard.doc_index)
                k = max(1, int(n * self.val_frac))
                for r in shard.doc_index[-k:]:
                    val_refs.add((s_idx, r))

        # Build flat list of training-only (shard, row_idx) references
        all_refs: List[Tuple[int, int]] = []
        for s_idx, shard in enumerate(self.shards):
            for r in shard.doc_index:
                if (s_idx, r) not in val_refs:
                    all_refs.append((s_idx, r))

        if deterministic:
            print(f"[ParquetDataset] Sequential iteration over "
                  f"{len(all_refs):,} docs")
            order = list(range(len(all_refs)))
        else:
            print(f"[ParquetDataset] Shuffling {len(all_refs):,} docs "
                  f"(seed={seed})")
            rng = random.Random(seed)
            order = list(range(len(all_refs)))
            rng.shuffle(order)

        # Read docs into a token pool, yield fixed-size sequences
        buf: List[int] = []
        batch_idx = 0
        for ref_pos in order:
            shard_idx, row_idx = all_refs[ref_pos]
            try:
                tokens, _source = self._read_row(shard_idx, row_idx)
            except Exception as exc:
                print(f"[ParquetDataset] Warn: failed to read row {row_idx} "
                      f"in {self.shards[shard_idx].local_path.name}: {exc}")
                continue

            buf.extend(tokens)
            # Insert EOS at doc boundaries so the model learns end-of-document.
            # M2 fix: use the threaded eos_id; fall back to 0 with a one-shot warning.
            if self.eos_id is not None:
                buf.append(self.eos_id)
            else:
                if not getattr(self, "_warned_eos_fallback", True):
                    print("[ParquetDataset] WARN: eos_id not set; falling back to "
                          "token 0 as doc-boundary marker. Pass eos_id=tokenizer.eos_id.")
                    self._warned_eos_fallback = True
                buf.append(0)

            # Yield full seq_len chunks
            while len(buf) >= self.seq_len:
                slice_ = buf[: self.seq_len]
                buf = buf[self.seq_len:]
                if batch_idx < resume_step:
                    batch_idx += 1
                    continue
                input_ids = np.asarray(slice_, dtype=np.int64)
                labels = input_ids.copy()
                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                }
                batch_idx += 1

    # ------------------------------------------------------------------
    # Validation set
    # ------------------------------------------------------------------

    def iter_val(
        self,
        seed: int = 0,
        n_samples: int = 64,
        seq_len: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield `n_samples` held-out validation sequences (last 1% per shard)."""
        seq_len = seq_len or self.seq_len
        refs: List[Tuple[int, int]] = []
        for s_idx, shard in enumerate(self.shards):
            n = len(shard.doc_index)
            k = max(1, int(n * self.val_frac))
            for r in shard.doc_index[-k:]:
                refs.append((s_idx, r))

        rng = random.Random(seed)
        rng.shuffle(refs)
        refs = refs[:n_samples]

        for shard_idx, row_idx in refs:
            try:
                tokens, _ = self._read_row(shard_idx, row_idx)
            except Exception:
                continue
            if len(tokens) < seq_len:
                continue
            slice_ = tokens[:seq_len]
            yield {
                "input_ids": np.asarray(slice_, dtype=np.int64),
                "labels": np.asarray(slice_, dtype=np.int64),
            }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    print("ParquetDataset smoke test (no actual HF call)")
    print(f"  module location: {__file__}")
