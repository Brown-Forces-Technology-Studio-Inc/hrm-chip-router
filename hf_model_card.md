---
language: en
license: apache-2.0
tags:
- chip-design
- routing
- EDA
- recursive-reasoning
- HRM
- bfts
pipeline_tag: token-classification
---

# HRM Chip Router: 7M Parameter Hierarchical Reasoning Model

The Brown Forces 7M HRM Chip Router is a research release of a Hierarchical Reasoning Model (HRM) trained to solve multi-net, multi-layer 3D chip routing problems. This model can route complex 3D grids in under 2 seconds on consumer hardware, demonstrating the power of recursive reasoning in electronic design automation (EDA).

## 🔧 Want higher connectivity on your real designs?

The 7M model here is the open research release. We're training a **35M enterprise model with RL-based reward** targeting >25% full-puzzle connectivity on production 3D layouts.

**→ [Join the waitlist at brownforces.io/chip-router](https://brownforces.io/chip-router)**

Enterprise routing for riscv32i / ibex / aes / jpeg-class designs. Multi-layer. Fast. No $100k Cadence license.

---

## Key Performance Metrics

*   **Token Accuracy:** 97.9% (spatial geometry fully transfers)
*   **Zero-Shot Transfer (ibex):** 87.6% connected, 0 obstacle violations
*   **Zero-Shot Transfer (riscv32i):** 65.1% connected, 0 obstacle violations
*   **Connectivity Solved-Rate (Full Puzzle):** 3.53%
*   **Connectivity Solved-Rate (Per Net):** ~25%
*   **Inference Speed:** ~2 seconds per 64-puzzle batch (RTX 5090)
*   **Model Size:** ~7M parameters (105 MB checkpoint)

## Architecture

The model uses a **Hierarchical Reasoning Model (HRM)** architecture with **Adaptive Computation Time (ACT)** halting. It operates on a 16×4×16 3D grid with a vocabulary of 7 tokens.

*   **Input Grid:** 16x4x16 (Row x Layer x Column)
*   **Vocabulary:** 7 tokens (0=pad, 1=empty, 2=obstacle, 3-6=net colors 0-3)
*   **Reasoning Steps:** 16 ACT steps per inference

## Usage

For full implementation and evaluation code, visit the [GitHub repository](https://github.com/Brown-Forces-Technology-Studio-Inc/hrm-chip-router).

### Quick Inference

```python
import torch
from evaluate_checkpoints import load_model_for_checkpoint, run_inference

# Load the 7M model (requires CUDA)
model, ds, meta = load_model_for_checkpoint("step_60764")
inputs, preds = run_inference(model, ds, meta, max_batches=1)

print(f"Routed {len(preds)} puzzles.")
```

## Citation

```bibtex
@article{trm2025,
  title={TinyRecursiveModels: Reasoning through Recursion},
  author={Samsung SAIL Montreal},
  journal={arXiv preprint arXiv:2510.04871},
  year={2025}
}

@misc{hrm_chip_router_2026,
  author = {Brown Forces Technology Studio Inc.},
  title = {HRM Chip Router: 7M Parameter Hierarchical Reasoning Model},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Brown-Forces-Technology-Studio-Inc/hrm-chip-router}}
}
```
