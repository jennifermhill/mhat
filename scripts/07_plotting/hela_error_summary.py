"""MHAT on Fluo-N2DL-HeLa: CTC metrics next to division error rates.

One axes, two groups of bars separated by a divider:

* left, "higher is better": TRA, DET, SEG (CTCMetrics from the run's
  track_metrics.json);
* right, "lower is better": FP divisions and FN divisions, each as a fraction
  of the total number of GT divisions (so they are on the same 0-1 axis as the
  CTC metrics; FP / GT can exceed 1 in principle).

Data provenance
---------------
* TRA / DET / SEG: the run's eval ``<output_dir>/<run>/evaluation/<eval>/track_metrics.json``.
* Divisions: recomputed live with traccuracy ``DivisionMetrics`` on the CTC
  matching (``match_tracking``) against ``<output_dir>/gt/ctc/``, at frame buffer ``--frame-buffer`` (default 1,
  the buffer used for BC(1) / BIO in hela_optimization_log.md). Two traccuracy
  0.4.3 workarounds, same as ``scratch_configs/experiments/hela_bio/bio.py``:
  ``override_matcher=True`` because the CTC matching is many-to-one, and pred
  nodes matched to several GT nodes (ns nodes) are dropped from the matching
  first, because the shifted-division correction raises on them.

Defaults are the held-out run (02_cells / hela_holdout_ft08) and its CTC eval;
for another run pass ``--output-dir``, ``--run`` and ``--eval``.

Usage:
    python scripts/07_plotting/hela_error_summary.py \
        [--output-dir experiments/Fluo-N2DL-HeLa/02_cells] [--run tracking/hela_holdout_ft08] \
        [--eval ctc_ctc_2026-09-24_15-10-46] [--output foo.png]
"""
import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from traccuracy import metrics
from traccuracy.matchers._matched import Matched

from mhat.evaluation.evaluate_tracking import match_tracking

CTC_METRICS = ["TRA", "DET", "SEG"]

# MHAT's CTC scores keep the pink used for MHAT in the other figures (Wong
# palette); the error rates get a lighter tint of the same pink, so every bar
# reads as MHAT and the two "directions" differ by more than position alone.
CTC_COLOR = "#CC79A7"    # reddish purple
ERROR_COLOR = "#E6BCD3"  # 50 % tint of CTC_COLOR toward white
INK = "#333333"
MUTED = "#666666"


def division_counts(output_dir, run, frame_buffer):
    """(total GT, FP, FN) divisions for one run at the given frame buffer."""
    gt_dir = output_dir / "gt" / "ctc"
    cfg = {"metrics": ["ctc"], "matcher": "ctc"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = match_tracking(
            cfg, gt_dir / "correct_tracks.zarr", gt_dir / "correct_seg.zarr", output_dir / run
        )
        n_gt_per_pred = Counter(p for _, p in m.mapping)
        one_to_one = Matched(
            m.gt_graph, m.pred_graph,
            [(g, p) for g, p in m.mapping if n_gt_per_pred[p] == 1],
            dict(m.matcher_info),
        )
        res = metrics.DivisionMetrics(max_frame_buffer=frame_buffer).compute(
            one_to_one, override_matcher=True
        ).results[f"Frame Buffer {frame_buffer}"]
    return (res["Total GT Divisions"], res["False Positive Divisions"],
            res["False Negative Divisions"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=Path("experiments/Fluo-N2DL-HeLa/02_cells"),
                    help="the HeLa movie's output directory")
    ap.add_argument("--run", default="tracking/hela_holdout_ft08",
                    help="the run, relative to --output-dir")
    ap.add_argument("--eval", default="ctc_ctc_2026-09-24_15-10-46",
                    help="the run's CTC eval (directory name)")
    ap.add_argument("--frame-buffer", type=int, default=1)
    ap.add_argument("--output", default=None, help="Override output PNG path")
    args = ap.parse_args()

    metrics_path = args.output_dir / args.run / "evaluation" / args.eval / "track_metrics.json"
    ctc = json.loads(metrics_path.read_text())["CTCMetrics"]
    n_gt, n_fp, n_fn = division_counts(args.output_dir, args.run, args.frame_buffer)
    dataset = args.output_dir.name
    print(f"{dataset}/{args.run}: " + ", ".join(f"{k} {ctc[k]:.4f}" for k in CTC_METRICS)
          + f"; GT divisions {n_gt}, FP {n_fp}, FN {n_fn} (frame buffer {args.frame_buffer})")

    labels = CTC_METRICS + ["FP\ndivisions", "FN\ndivisions"]
    values = [ctc[k] for k in CTC_METRICS] + [n_fp / n_gt, n_fn / n_gt]
    annotations = ([f"{v:.3f}" for v in values[:3]]
                   + [f"{v:.2f}\n({n}/{n_gt})" for v, n in zip(values[3:], (n_fp, n_fn))])
    colors = [CTC_COLOR] * 3 + [ERROR_COLOR] * 2
    # A one-bar gap between the groups holds the divider.
    x = np.array([0, 1, 2, 4, 5], dtype=float)

    fig, ax = plt.subplots(figsize=(5.8, 4))
    bars = ax.bar(x, values, 0.72, color=colors)
    for bar, text in zip(bars, annotations):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, text,
                ha="center", va="bottom", fontsize=8, color=INK)

    ax.axvline(3, color="#bbbbbb", linewidth=1.2)
    ymax = 1.12
    for xc, text in ((1, "higher is better"), (4.5, "lower is better")):
        ax.text(xc, ymax - 0.01, text, ha="center", va="top", fontsize=9,
                color=MUTED, style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Score / fraction of GT divisions", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = Path(args.output or args.output_dir / "figures" / f"hela_error_summary_{dataset}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
