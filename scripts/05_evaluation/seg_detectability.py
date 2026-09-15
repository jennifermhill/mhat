"""Score a segmentation run by whether its candidate pool can detect the GT markers.

Answers the question the tracking diagnostics cannot: for each CTC ground-truth
marker, is there *any* hypothesis in the segmentation the tracker could have
matched it to? See ``mhat.evaluation.seg_detectability`` for the criterion and
for how the three candidate pools it reports differ.

    conda run -n mhat-cluster --no-capture-output python -u \\
        scripts/05_evaluation/seg_detectability.py \\
        --seg-dir  .../segmentation/Fluo-N3DL-DRO/01_nuclei_short/<uid> \\
        --gt-tra   .../tracking/Fluo-N3DL-DRO/01_nuclei/01_GT/TRA

``--size-threshold`` / ``--min-cost`` / ``--max-cost`` must mirror the *tracking*
config the segmentation is destined for, since they define the pool the solver
sees. They default to the Fluo-N3DL-DRO operating point.

``--baseline`` takes another run's ``detectability.json`` and adds a per-marker
comparison, which is the number that actually matters when sweeping: how many
markers that were undetectable before are detectable now, and how many
regressed the other way.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mhat.evaluation.seg_detectability import (
    CTC_DETECTION_FRACTION,
    evaluate,
    summarize,
    write_results,
)


def compare(per_frame, baseline_path):
    """Per-marker gained/lost against a baseline run's detectability.json."""
    with open(baseline_path) as handle:
        baseline = json.load(handle)
    base_frames = baseline["per_frame"]

    gained, lost, common = [], [], 0
    for t, frame in enumerate(per_frame):
        if t >= len(base_frames):
            break
        base = base_frames[t]
        for marker, entry in frame.items():
            base_entry = base.get(str(marker))
            if base_entry is None:
                continue
            common += 1
            now = entry["gated"] > CTC_DETECTION_FRACTION
            before = base_entry["gated"] > CTC_DETECTION_FRACTION
            if now and not before:
                gained.append({"t": t, "marker": marker,
                               "was": base_entry["gated"], "now": entry["gated"]})
            elif before and not now:
                lost.append({"t": t, "marker": marker,
                             "was": base_entry["gated"], "now": entry["gated"]})

    return {
        "baseline": baseline.get("label", str(baseline_path)),
        "markers_compared": common,
        "gained": len(gained),
        "lost": len(lost),
        "net": len(gained) - len(lost),
        "gained_detail": gained,
        "lost_detail": lost,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seg-dir", required=True,
                        help="segmentation run dir (holds data.zarr + merge_history.csv)")
    parser.add_argument("--gt-tra", required=True, help="CTC GT TRA dir of man_track*.tif")
    parser.add_argument("--n-frames", type=int, default=None,
                        help="limit to the first N frames (default: all in the run)")
    parser.add_argument("--size-threshold", type=int, default=20)
    parser.add_argument("--min-cost", type=float, default=0.0)
    parser.add_argument("--max-cost", type=float, default=1.0)
    parser.add_argument("--out-dir", default=None,
                        help="default: <seg-dir>/detectability")
    parser.add_argument("--label", default=None, help="default: the seg dir name")
    parser.add_argument("--baseline", default=None,
                        help="another run's detectability.json, for a gained/lost diff")
    args = parser.parse_args()

    seg_dir = Path(args.seg_dir)
    label = args.label or seg_dir.name
    out_dir = Path(args.out_dir) if args.out_dir else seg_dir / "detectability"

    print(f"Scoring {seg_dir}")
    per_frame = evaluate(
        seg_dir, args.gt_tra,
        n_frames=args.n_frames,
        size_threshold=args.size_threshold,
        min_cost=args.min_cost,
        max_cost=args.max_cost,
    )
    summary = summarize(per_frame)
    report = write_results(out_dir, label, per_frame, summary)
    print()
    print(report)

    if args.baseline:
        diff = compare(per_frame, args.baseline)
        with open(out_dir / "detectability_vs_baseline.json", "w") as handle:
            json.dump(diff, handle)
        print()
        print(f"vs baseline {diff['baseline']}: "
              f"+{diff['gained']} / -{diff['lost']} = net {diff['net']:+d} "
              f"over {diff['markers_compared']} markers")

    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
