"""Merge several GT-amount result CSVs into one, tagging each with a condition.

Lets a single learning-curve figure show more than one sweep — e.g. the first pass
next to a second pass at a different regularization. Each input's `arm` values are
rewritten to `<arm>|<label>` so the plotting script treats them as separate series.

`full` rows (the all-tracks run) are **not** shared between conditions. Whether the
full-GT fit is identical across sweeps depends on the knob being varied — it is under
`ssvm_reg_normalize` (normalization is exactly 1.0 at full GT) but not under an
explicit `ssvm_reg_effective_target`, where the full-GT run gets its own rescaled
ssvm_reg. Rather than guess, each condition's full row is copied into every arm of
that condition, so each curve carries its own endpoint and nothing is silently
dropped.

Usage:
    python scripts/05_evaluation/merge_gt_amount_csvs.py merged.csv \
        pass1=gt_amount_train_results.csv pass2=gt_amount_reg261_train_results.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", nargs="+", help="LABEL=path/to/results.csv")
    args = parser.parse_args()

    merged = []
    fieldnames: list[str] = []

    for spec in args.inputs:
        if "=" not in spec:
            parser.error(f"expected LABEL=path, got {spec!r}")
        label, path = spec.split("=", 1)
        rows = list(csv.DictReader(open(path, newline="")))
        if not rows:
            parser.error(f"{path} has no rows")
        for key in rows[0]:
            if key not in fieldnames:
                fieldnames.append(key)
        for key in ("condition",):
            if key not in fieldnames:
                fieldnames.append(key)

        arms = sorted({r["arm"] for r in rows if r["arm"] != "full"})
        n_full = 0
        for row in rows:
            row["condition"] = label
            if row["arm"] == "full":
                # give every arm of this condition its own copy of the endpoint
                for arm in arms:
                    copy = dict(row)
                    copy["arm"] = f"{arm}|{label}"
                    merged.append(copy)
                n_full += 1
                continue
            row["arm"] = f"{row['arm']}|{label}"
            merged.append(row)
        print(f"{label}: {len(rows)} rows, {n_full} full-GT row(s) expanded across {arms}")

    merged.sort(key=lambda r: (r["arm"], -int(r["n_tracks"]), int(r["seed"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)
    print(f"\nWrote {len(merged)} rows to {args.output}")
    print("arms:", ", ".join(sorted({r["arm"] for r in merged})))


if __name__ == "__main__":
    main()
