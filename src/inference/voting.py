"""Majority voting and best-of-N for math reasoning.

Per /ipho-physicist: sampling N independent solutions and taking the
majority answer is a powerful way to improve accuracy without
increasing model parameters (Wang et al. 2023, "Self-Consistency").

This module provides:
- `majority_vote`: Sample N, take the most common answer
- `best_of_n`: Sample N, take the highest-reward answer
- `aggregate_metrics`: Compute agreement statistics
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

from src.inference.generation import RWKVGenerator
from src.training.reward import compute_reward
from src.utils.math_verify import extract_boxed, extract_number


def majority_vote(
    candidates: List[str],
    fallback_gold: Optional[str] = None,
) -> Tuple[Optional[str], Dict]:
    """Take the most common answer among candidates.

    Args:
        candidates: List of generated responses.
        fallback_gold: If no agreement, return this.

    Returns:
        (majority_answer, info_dict)
    """
    if not candidates:
        return fallback_gold, {"agree_count": 0}

    # Extract answer from each candidate; fall back to the raw candidate string
    # (useful when the caller already provides clean answer strings).
    answers = []
    for c in candidates:
        ans = extract_boxed(c) or extract_number(c)
        if ans is None:
            ans = c.strip()
        answers.append(ans)

    # Count (after de-duping identical strings/None entries)
    counter = Counter(a for a in answers if a)
    if not counter:
        return fallback_gold, {"agree_count": 0}

    top_answer, top_count = counter.most_common(1)[0]
    info = {
        "agree_count": top_count,
        "total": len(candidates),
        "agreement_rate": top_count / len(candidates),
        "all_answers": dict(counter),
    }
    return top_answer, info


def best_of_n(
    candidates: List[str],
    gold: Optional[str] = None,
) -> Tuple[Optional[str], Dict]:
    """Pick the best candidate based on reward.

    If gold is provided, uses reward_correctness.
    Otherwise uses reward_format + reward_length.

    Args:
        candidates: List of generated responses.
        gold: Gold answer (if known, for correctness reward).

    Returns:
        (best_answer, info_dict)
    """
    if not candidates:
        return None, {"reward": 0.0}

    rewards = []
    for c in candidates:
        if gold is not None:
            r = compute_reward(c, gold)
        else:
            # Use format + length only
            from src.training.reward import reward_format, reward_length
            r = 0.3 * reward_format(c) + 0.2 * reward_length(c, "")
        rewards.append(r)

    best_idx = max(range(len(candidates)), key=lambda i: rewards[i])
    best = candidates[best_idx]
    ans = extract_boxed(best) or extract_number(best)
    info = {
        "reward": rewards[best_idx],
        "best_idx": best_idx,
        "all_rewards": rewards,
    }
    return ans, info


def sample_and_vote(
    model,
    tokenizer,
    prompt: str,
    n_samples: int = 16,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    max_new_tokens: int = 256,
    method: str = "majority",
    gold: Optional[str] = None,
) -> Tuple[Optional[str], Dict]:
    """Sample N candidates and aggregate.

    Args:
        model: RWKV-7 model
        tokenizer: Tokenizer
        prompt: Input prompt
        n_samples: Number of candidates to sample
        temperature, top_p, top_k: Sampling params
        max_new_tokens: Max generation length per sample
        method: "majority" or "best_of_n"
        gold: Reference answer (for best_of_n)

    Returns:
        (selected_answer, info_dict)
    """
    gen = RWKVGenerator(model, tokenizer)
    candidates = gen.generate_batch(
        prompts=[prompt] * n_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    if method == "majority":
        ans, info = majority_vote(candidates)
    elif method == "best_of_n":
        ans, info = best_of_n(candidates, gold=gold)
    else:
        raise ValueError(f"Unknown method: {method}")

    info["candidates"] = candidates
    info["n_samples"] = n_samples
    return ans, info


def aggregate_metrics(
    results: List[Dict],
) -> Dict:
    """Aggregate metrics across multiple voting runs.

    Args:
        results: List of dicts with "correct" (bool) and other info.

    Returns:
        Aggregated metrics: pass@1, pass@n, agreement_rate, etc.
    """
    if not results:
        return {}

    total = len(results)
    correct = sum(1 for r in results if r.get("correct", False))
    agreements = [r.get("agreement_rate", 0.0) for r in results]

    return {
        "total": total,
        "accuracy": correct / total,
        "pass_at_1": correct / total,
        "mean_agreement": sum(agreements) / len(agreements) if agreements else 0.0,
    }


if __name__ == "__main__":
    # Smoke test
    candidates = [
        "<REASON>2+2 =4</REASON><ANSWER>4</ANSWER>",
        "<REASON>2+2 =4</REASON><ANSWER>4</ANSWER>",
        "<REASON>2+2 =5</REASON><ANSWER>5</ANSWER>",
        "<REASON>2+2 =4</REASON><ANSWER>4</ANSWER>",
    ]
    ans, info = majority_vote(candidates)
    print(f"Majority: {ans}, info: {info}")

    ans, info = best_of_n(candidates, gold="4")
    print(f"Best-of-N: {ans}, info: {info}")