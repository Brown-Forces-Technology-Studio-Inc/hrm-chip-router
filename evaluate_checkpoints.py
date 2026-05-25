"""
evaluate_checkpoints.py
Evaluates each saved Route3D-rich checkpoint on the test set,
computing connectivity_solved_rate and optionally visualizing sample predictions.

Usage:
  python evaluate_checkpoints.py \
      --metric connectivity_solved_rate \
      --visualize true
"""
import argparse, os, sys, json, time
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# ── paths ────────────────────────────────────────────────────────────────────
# These can be overridden by environment variables
REPO     = os.environ.get("HRM_REPO", os.getcwd())
CKPT_DIR = os.environ.get("HRM_CKPT_DIR", os.path.join(REPO, "checkpoints"))
DATA_DIR = os.environ.get("HRM_DATA_DIR", os.path.join(REPO, "data"))
OUT_DIR  = os.environ.get("HRM_OUT_DIR", os.path.join(REPO, "eval_results"))
sys.path.insert(0, REPO)

# ── token meaning ─────────────────────────────────────────────────────────────
# ... (rest of the file) ...
# vocab_size = 7 (tokens 0-6)
#   0 = pad
#   1 = empty
#   2 = obstacle
#   3,4,5,6 = net 0-3 wire (terminals + routed wire in label)
NET_TOKENS = [3, 4, 5, 6]          # 4 nets for k=4
GRID_ROWS, GRID_LAYERS, GRID_COLS = 16, 4, 16   # 16*4*16 = 1024

# 3D 6-connectivity neighbours: (dr, dl, dc)
_NEIGHBOURS = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

# ── connectivity check ────────────────────────────────────────────────────────
def flat_to_3d(idx: int) -> Tuple[int,int,int]:
    c = idx % GRID_COLS
    idx //= GRID_COLS
    l = idx % GRID_LAYERS
    r = idx // GRID_LAYERS
    return r, l, c

def _3d_to_flat(r, l, c) -> int:
    return r * GRID_LAYERS * GRID_COLS + l * GRID_COLS + c

def connectivity_solved(inp_flat: np.ndarray, pred_flat: np.ndarray) -> Dict[int, bool]:
    """
    For each net token t in NET_TOKENS:
      - terminals = positions where inp_flat == t
      - wire      = positions where pred_flat == t
      Returns True if all terminals are connected through the predicted wire
      (BFS over pred cells with token t, checking that all terminal positions
       are reachable from the first terminal).
    Skips nets that have fewer than 2 terminals in the input.
    """
    results = {}
    inp_3d  = inp_flat.reshape(GRID_ROWS, GRID_LAYERS, GRID_COLS)
    pred_3d = pred_flat.reshape(GRID_ROWS, GRID_LAYERS, GRID_COLS)

    for t in NET_TOKENS:
        term_positions = list(zip(*np.where(inp_3d == t)))
        if len(term_positions) < 2:
            # no net present in this puzzle, skip
            continue

        # BFS from first terminal, traversing cells where pred == t
        start = term_positions[0]
        visited = set()
        queue = deque([start])
        visited.add(start)

        while queue:
            r, l, c = queue.popleft()
            for dr, dl, dc in _NEIGHBOURS:
                nr, nl, nc = r+dr, l+dl, c+dc
                if 0 <= nr < GRID_ROWS and 0 <= nl < GRID_LAYERS and 0 <= nc < GRID_COLS:
                    nb = (nr, nl, nc)
                    if nb not in visited and pred_3d[nr, nl, nc] == t:
                        visited.add(nb)
                        queue.append(nb)

        # All terminals must be in visited
        solved = all(tp in visited for tp in term_positions)
        results[t] = solved

    return results


def compute_connectivity_solved_rate(inputs: np.ndarray, preds: np.ndarray) -> Dict:
    """
    Inputs/preds: (N, 1024) uint8
    Returns per-net solved rates and overall mean.
    """
    per_net = {t: [] for t in NET_TOKENS}
    exact = []

    for i in range(len(inputs)):
        r = connectivity_solved(inputs[i], preds[i])
        all_solved = True
        for t, solved in r.items():
            per_net[t].append(float(solved))
            if not solved:
                all_solved = False
        if r:  # puzzle had at least one net
            exact.append(float(all_solved))

    summary = {
        f"net_{t}_solved_rate": float(np.mean(per_net[t])) if per_net[t] else float("nan")
        for t in NET_TOKENS
    }
    summary["connectivity_solved_rate"] = float(np.mean(exact)) if exact else float("nan")
    summary["exact_route_solved_rate"]  = summary["connectivity_solved_rate"]
    summary["puzzles_evaluated"] = len(exact)
    return summary


# ── model loading ─────────────────────────────────────────────────────────────
def load_model_for_checkpoint(ckpt_path: str):
    from utils.functions import load_model_class
    from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig

    # Try all_config.yaml in ckpt directory, then in REPO root
    config_path = os.path.join(os.path.dirname(ckpt_path), "all_config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(REPO, "all_config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find all_config.yaml at {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Load dataset metadata
    ds_cfg = PuzzleDatasetConfig(
        seed=0,
        dataset_paths=[DATA_DIR],
        global_batch_size=cfg["global_batch_size"],
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )
    ds = PuzzleDataset(ds_cfg, split="test")
    ds._lazy_load_dataset()
    meta = ds.metadata

    arch = cfg["arch"]
    model_cfg = dict(
        **{k: v for k, v in arch.items() if k not in ("name", "loss")},
        batch_size=cfg["global_batch_size"],
        vocab_size=meta.vocab_size,
        seq_len=meta.seq_len,
        num_puzzle_identifiers=meta.num_puzzle_identifiers,
        causal=False,
    )

    model_cls   = load_model_class(arch["name"])
    loss_cls    = load_model_class(arch["loss"]["name"])

    with torch.device("cuda"):
        model = model_cls(model_cfg)
        model = loss_cls(model, **{k: v for k, v in arch["loss"].items() if k != "name"})
        model = torch.compile(model)

    print(f"  Loading weights from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cuda")
    # compiled model wraps with _orig_mod prefix sometimes
    try:
        model.load_state_dict(state_dict, assign=True)
    except RuntimeError:
        # strip _orig_mod. prefix if present
        new_sd = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_sd, assign=True)

    model.eval()
    return model, ds, meta


# ── inference ─────────────────────────────────────────────────────────────────
def run_inference(model, ds, meta, max_batches: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (inputs_np, preds_np) both shape (N,1024) uint8"""
    from models.losses import IGNORE_LABEL_ID

    all_inputs = []
    all_preds  = []
    batch_idx  = 0

    with torch.inference_mode():
        for set_name, batch, global_bs in ds:
            batch = {k: v.cuda() for k, v in batch.items()}

            with torch.device("cuda"):
                carry = model.initial_carry(batch)

            steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = model(
                    carry=carry, batch=batch, return_keys={"preds"}
                )
                steps += 1
                if all_finish:
                    break

            # global_bs is the real (unpadded) count for this batch
            # puzzle_identifiers can't be used as mask when blank_identifier_id==0
            # and all route3d puzzles also have identifier 0
            n = global_bs  # real sample count in this batch
            inp  = batch["inputs"][:n].cpu().numpy().astype(np.uint8)
            pred = preds["preds"][:n].cpu().numpy().astype(np.uint8)

            all_inputs.append(inp)
            all_preds.append(pred)
            batch_idx += 1
            print(f"    batch {batch_idx}: {len(inp)} samples, {steps} inference steps")

            if max_batches and batch_idx >= max_batches:
                break

    return np.concatenate(all_inputs), np.concatenate(all_preds)


# ── visualization ─────────────────────────────────────────────────────────────
TOKEN_COLORS = {
    0: (1,1,1,0),       # pad — transparent
    1: (0.92,0.92,0.92,1),  # empty
    2: (0.2,0.2,0.2,1),    # obstacle
    3: (0.2,0.6,1.0,1),    # net 0 — blue
    4: (0.9,0.3,0.3,1),    # net 1 — red
    5: (0.2,0.8,0.3,1),    # net 2 — green
    6: (0.9,0.7,0.1,1),    # net 3 — yellow
}
TOKEN_LABELS = {1:"empty",2:"obstacle",3:"net-0",4:"net-1",5:"net-2",6:"net-3"}


def visualize_sample(inp_flat, pred_flat, label_flat, puzzle_idx: int, step: int, out_dir: str):
    inp_3d  = inp_flat.reshape(GRID_ROWS, GRID_LAYERS, GRID_COLS)
    pred_3d = pred_flat.reshape(GRID_ROWS, GRID_LAYERS, GRID_COLS)
    lab_3d  = label_flat.reshape(GRID_ROWS, GRID_LAYERS, GRID_COLS)

    def grid_to_rgba(g):
        rgba = np.zeros((*g.shape, 4))
        for tok, col in TOKEN_COLORS.items():
            rgba[g == tok] = col
        return rgba

    fig, axes = plt.subplots(3, GRID_LAYERS, figsize=(GRID_LAYERS*4, 10))
    fig.suptitle(f"Puzzle {puzzle_idx} — checkpoint step_{step}", fontsize=14)

    for l in range(GRID_LAYERS):
        axes[0,l].imshow(grid_to_rgba(inp_3d[:,l,:]), aspect="equal", interpolation="nearest")
        axes[0,l].set_title(f"Input layer {l}")
        axes[1,l].imshow(grid_to_rgba(pred_3d[:,l,:]), aspect="equal", interpolation="nearest")
        axes[1,l].set_title(f"Pred layer {l}")
        axes[2,l].imshow(grid_to_rgba(lab_3d[:,l,:]), aspect="equal", interpolation="nearest")
        axes[2,l].set_title(f"Label layer {l}")
        for ax in axes[:,l]: ax.axis("off")

    legend_patches = [Patch(color=TOKEN_COLORS[t][:3], label=lbl)
                      for t, lbl in TOKEN_LABELS.items()]
    fig.legend(handles=legend_patches, loc="lower center", ncol=len(legend_patches), fontsize=9)
    plt.tight_layout(rect=[0,0.04,1,1])

    path = os.path.join(out_dir, f"step_{step}_puzzle_{puzzle_idx}.png")
    plt.savefig(path, dpi=100)
    plt.close(fig)
    print(f"    Saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric",    default="connectivity_solved_rate")
    ap.add_argument("--visualize", default="false")
    ap.add_argument("--max_batches", type=int, default=None,
                    help="Limit batches per checkpoint (for quick smoke test)")
    ap.add_argument("--vis_samples", type=int, default=6,
                    help="Number of sample puzzles to visualize per checkpoint")
    args = ap.parse_args()
    do_vis = args.visualize.lower() in ("true","1","yes")

    os.makedirs(OUT_DIR, exist_ok=True)

    # find checkpoints
    ckpts = sorted(
        [f for f in os.listdir(CKPT_DIR) if f.startswith("step_")],
        key=lambda x: int(x.split("_")[1])
    )
    print(f"Checkpoints found: {ckpts}")

    # also load labels for visualization
    labels_np = np.load(f"{DATA_DIR}/test/all__labels.npy", mmap_mode="r")

    all_results = {}

    for ckpt_name in ckpts:
        step = int(ckpt_name.split("_")[1])
        ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
        print(f"\n{'='*60}")
        print(f"Evaluating {ckpt_name} ...")
        t0 = time.time()

        model, ds, meta = load_model_for_checkpoint(ckpt_path)

        inputs_np, preds_np = run_inference(model, ds, meta, max_batches=args.max_batches)
        print(f"  Inference done: {len(inputs_np)} samples in {time.time()-t0:.1f}s")

        # compute metric
        results = compute_connectivity_solved_rate(inputs_np, preds_np)
        results["checkpoint"] = ckpt_name
        results["eval_time_s"] = round(time.time() - t0, 1)
        all_results[ckpt_name] = results

        print(f"\n  ── Results for {ckpt_name} ──")
        for k, v in results.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

        # visualization
        if do_vis:
            # pick solved and unsolved samples for contrast
            solved_idx   = []
            unsolved_idx = []
            for i in range(len(inputs_np)):
                r = connectivity_solved(inputs_np[i], preds_np[i])
                if r and all(r.values()):
                    solved_idx.append(i)
                elif r:
                    unsolved_idx.append(i)

            sample_ids = (solved_idx[:args.vis_samples//2] +
                          unsolved_idx[:args.vis_samples - len(solved_idx[:args.vis_samples//2])])
            print(f"  Visualizing {len(sample_ids)} samples "
                  f"({len(solved_idx[:args.vis_samples//2])} solved, "
                  f"{len(unsolved_idx[:args.vis_samples - len(solved_idx[:args.vis_samples//2])])} unsolved)")
            for i in sample_ids:
                visualize_sample(inputs_np[i], preds_np[i], labels_np[i],
                                 puzzle_idx=i, step=step, out_dir=OUT_DIR)

        del model
        torch.cuda.empty_cache()

    # summary table
    print(f"\n{'='*60}")
    print("SUMMARY — connectivity_solved_rate by checkpoint:")
    print(f"  {'Checkpoint':<20} {'conn_solved_rate':>18} {'puzzles':>10} {'time(s)':>10}")
    for ckpt_name, r in all_results.items():
        print(f"  {ckpt_name:<20} {r['connectivity_solved_rate']:>18.4f} "
              f"{r['puzzles_evaluated']:>10} {r['eval_time_s']:>10.1f}")

    # save JSON
    out_json = os.path.join(OUT_DIR, "eval_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_json}")


if __name__ == "__main__":
    main()
