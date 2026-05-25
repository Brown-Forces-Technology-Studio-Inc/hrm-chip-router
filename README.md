# HRM Chip Router: 7M Parameter Hierarchical Reasoning Model

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange)](https://huggingface.co/salvadordabrown/hrm-chip-router-7m)

The Brown Forces 7M HRM Chip Router is a research release of a Hierarchical Reasoning Model (HRM) trained to solve multi-net, multi-layer 3D chip routing problems. This model can route complex 3D grids in under 2 seconds on consumer hardware, demonstrating the power of recursive reasoning in electronic design automation (EDA).

## 🔧 Want production-grade routing on your real designs?

This is the open research release. We're building an enterprise version for production-scale designs — ibex, riscv32i, aes, jpeg-class and beyond. Multi-layer. Fast. No $100k Cadence license.

**→ [Join the waitlist at brownforces.io/chip-router](https://brownforces.io/chip-router)**

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

## Sample Visualizations

Below are sample routing results from the 7M model showing input terminals and obstacles versus the predicted routes.

![Sample Routing 0](assets/step_60764_puzzle_0.png)
![Sample Routing 1](assets/step_60764_puzzle_1.png)

## Quick Start

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Brown-Forces-Technology-Studio-Inc/hrm-chip-router.git
    cd hrm-chip-router
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Run Inference

Download the checkpoint from [Hugging Face](https://huggingface.co/salvadordabrown/hrm-chip-router-7m) and place it in a `checkpoints/` directory.

```python
import torch
from evaluate_checkpoints import load_model_for_checkpoint, run_inference

# Load the 7M model (requires CUDA)
model, ds, meta = load_model_for_checkpoint("checkpoints/pytorch_model.bin")
inputs, preds = run_inference(model, ds, meta, max_batches=1)

print(f"Routed {len(preds)} puzzles.")
```

## Citation

If you use this work in your research, please cite the original TinyRecursiveModels paper and our HRM implementation:

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

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

---
*For more information, visit [brownforces.io](https://brownforces.io).*
