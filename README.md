# RWKV-7 50M Math LLM

A ~50M parameter RWKV-7 (Goose) language model for high-school math reasoning.
CPU-optimized architecture with state-space recurrence, trained via pretrain → SFT → GRPO.

**Key numbers:**
- **44.4M params** (well within the 50M target)
- **8 layers**, d_model=512, d_state=64 (8 heads)
- **4096 ctx_len**, math-aware digit-by-digit tokenizer
- **GRPO** with MC-GRPO baseline (median), PPO clipping, forward KL

## Quick Start

### Local CPU (tests only)
```bash
# Run full test suite
python -m pytest tests/ -v

# Smoke test: end-to-end pipeline
python -m pytest tests/test_pipeline.py -v

# Param count verification
python scripts/verify_param_count.py
```

### Modal A100 (full training)
```bash
# 1. Prepare data
modal run modal_app.py --stage pretrain --batch-size 16 --max-steps 50000

# 2. SFT
modal run modal_app.py --stage sft --batch-size 8 --max-steps 5000

# 3. GRPO
modal run modal_app.py --stage grpo --group-size 8 --max-steps 3000

# 4. Evaluate
modal run modal_app.py --stage eval --ckpt-path /checkpoints/grpo/final.pt
```

## Architecture

```
RWKV7Model
├── Embedding (32768 → 512)
├── RWKV7Block × 8
│   ├── RWKV7TimeMix (RWKV-7 WKV operator + LoRA decay/aaa/mv/gate)
│   │   ├── wkv7_op: state = state * w + state @ (a⊗b) + v⊗k
│   │   └── output = GroupNorm → r*k boost → gate projection
│   └── RWKV7ChannelMix (squared-ReLU FFN, dim_ffn=2048)
├── LayerNorm
└── Output head (tied to embedding)
```

**Param breakdown (44,382,208 total):**
- Embedding: 16,777,216
- 8× RWKV7Block: 27,602,944
- ln_out: 1,024

## Training Pipeline

### 1. Pretrain (~50K steps, A100 ~3h)
Standard next-token prediction on arithmetic + GSM8K + MATH text.
```
LR: 6e-4, cosine schedule, warmup=1000
Batch: 16 × 4096 ctx → ~4.7B tokens/epoch
```

### 2. SFT (~5K steps)
Format: `Question: ... Answer: <REASON>...CoT...</REASON><ANSWER>n</ANSWER>`
```
LR: 1e-5, warmup=100
Batch: 8 × 2048 ctx
```

### 3. GRPO — *optional, gated experiment*
```
G=8 rollouts/prompt → median baseline (robust, low variance)
PPO clip ε=0.2, forward KL β=0.04
```
Reward = correctness(1.0) + format(0.3) + length(0.2)

**NOT a default pipeline stage.** Per the 135M GRPO ablation finding
(GSM8K accuracy decreased under GRPO at this scale, not just late-stage),
we only run GRPO if SFT alone already shows real signal. Gate criteria:
  - SFT GSM8K pass@1 ≥ 25% AND SFT val loss < 2.3
  - AND a held-out SFT-vs-SFT+GRPO A/B comparison passes a paired
    bootstrap test (p<0.05, ≥200 prompts)
If those gates fail, skip GRPO entirely and ship the SFT model.

## Inference

```python
from src.inference.generation import RWKVGenerator
from src.inference.voting import sample_and_vote

gen = RWKVGenerator(model, tokenizer)

# Single sample
text = gen.generate("What is 123 + 456?", max_new_tokens=256, greedy=True)

# Majority vote (16 samples)
ans, info = sample_and_vote(model, tokenizer, prompt, n_samples=16)
```

## Evaluation

```bash
# Arithmetic (fast, no download)
python scripts/evaluate.py arithmetic --checkpoint checkpoints/grpo/final.pt --n-examples 100

# GSM8K
python scripts/evaluate.py gsm8k --checkpoint checkpoints/grpo/final.pt --split test

# MATH
python scripts/evaluate.py math --checkpoint checkpoints/grpo/final.pt --split test
```

## Project Structure

```
a:\resonanc\
├── src/
│   ├── model/          # config.py, rwkv7.py, init.py
│   ├── tokenizer/       # math_tokenizer.py (digit-by-digit BPE)
│   ├── data/           # gsm8k.py, math_dataset.py, synthetic.py, collator.py
│   ├── training/       # optimizer.py, pretrain.py, sft.py, grpo.py, reward.py
│   ├── inference/      # generation.py, voting.py
│   ├── eval/           # gsm8k_eval.py, math_eval.py, arithmetic_eval.py
│   └── utils/          # math_verify.py, seed.py
├── tests/              # test_*.py (all pytest, CPU-runnable)
├── scripts/             # prepare_data.py, train.py, evaluate.py
├── modal_app.py         # Modal A100 deployment
├── config.yaml          # All hyperparameters
├── requirements.txt
└── README.md
```

## Testing

| Test file | Coverage |
|-----------|----------|
| `test_model_dims.py` | Param count, shapes, gradients |
| `test_model_math.py` | Decay bounds, softmax, WKV causality, FD gradient check |
| `test_tokenizer.py` | Roundtrip, digit-split, special tokens |
| `test_grpo.py` | MC-GRPO median, PPO clipping, KL direction |
| `test_reward.py` | Correctness, format, length |
| `test_inference.py` | Determinism, batch, EOS, state continuity |
| `test_pipeline.py` | End-to-end pretrain→SFT→GRPO→eval |

Run: `python -m pytest tests/ -v --tb=short`

## RWKV-7 Reference

Architecture from [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM), cloned to `A:\rwkv_lm_ref`. The canonical `RWKV-v7/rwkv_v7_demo.py` was used as reference for:
- `rwkv7.py`: WKV operator, TimeMix, ChannelMix, Block
- `init.py`: Orthogonal init for projections, 0.5 for time-shift scalars
- State update: `state = state * w + state @ (a⊗b) + v⊗k`

## Verification Discipline

Every mathematical property is unit-tested:
- Decay `w = exp(-exp(w_raw))` ∈ (0,1) in safe range
- `ln_x` GroupNorm uses eps=64e-5
- MC-GRPO median baseline: `sorted[G//2]` (odd) or `(sorted[G//2-1]+sorted[G//2])/2` (even)
- Forward KL `KL[πθ ‖ πref]`: prevents mode collapse
- Digit-by-digit tokenizer: no token encodes 2+ digits
