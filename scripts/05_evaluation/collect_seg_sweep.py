"""Read a segmentation sweep's detectability results into one table.

Reads the manifest ``launch_seg_sweep.py`` wrote, pulls each run's
``detectability.json``, and writes ``results.csv`` plus a ranked report next to
the manifest.

**gated recall** -- the fraction of GT markers some candidate in the tracker's
pool could detect -- is the ceiling on DET, so no amount of ILP tuning can
exceed it. On Fluo-N3DL-DRO it is also nearly saturated (0.9915 at baseline),
which is why runs are ranked on ``over_merge_rate`` instead -- the fraction of
markers whose smallest covering candidate is too big to be one cell. Its older
sibling ``collisions`` (markers forced to share
one candidate are links ``MaxParents(1)`` forbids outright, and they were 64 of
71 lineage truncations) is kept as a secondary column, but it only fires when
*two annotated* markers claim one candidate and so misses most fusions on
sparsely annotated data. Recall is the constraint, over-merge is the objective.
``offset`` is the tiebreaker -- a candidate whose centre sits far from the
nucleus makes the correct link cost more than a wrong one.

Against the sweep's ``baseline`` run it also reports per-marker ``gained`` and
``lost``. Net recall can be flat while both are large, which means the run
traded one set of markers for another rather than improving; the counts catch
that where the recall alone does not.

    conda run -n mhat-cluster --no-capture-output python -u \\
        scripts/05_evaluation/collect_seg_sweep.py <manifest.toml>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import toml

from mhat.evaluation.seg_detectability import CTC_DETECTION_FRACTION


def load_run(record):
    path = Path(record["seg_dir"]) / "detectability" / "detectability.json"
    if not path.is_file():
        return None
    with open(path) as handle:
        return json.load(handle)


def detected_set(payload):
    """(frame, marker) pairs the tracker's pool can detect."""
    return {
        (t, marker)
        for t, frame in enumerate(payload["per_frame"])
        for marker, entry in frame.items()
        if entry["gated"] > CTC_DETECTION_FRACTION
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("manifest")
    args = parser.parse_args()

    manifest = toml.load(args.manifest)
    sweep_dir = Path(manifest["sweep_dir"])
    baseline_label = manifest.get("baseline", "")

    loaded = {}
    missing = []
    for record in manifest["runs"]:
        payload = load_run(record)
        if payload is None:
            missing.append(record["label"])
        else:
            loaded[record["label"]] = payload

    if not loaded:
        raise SystemExit("no detectability.json found for any run")

    base_detected = (
        detected_set(loaded[baseline_label]) if baseline_label in loaded else None
    )

    rows = []
    for record in manifest["runs"]:
        payload = loaded.get(record["label"])
        if payload is None:
            continue
        summary = payload["summary"]
        row = {
            "label": record["label"],
            "exp_uid": record["exp_uid"],
            "recall_gated": round(summary["recall"]["gated"], 4),
            "collided": summary.get("n_markers_collided", ""),
            "collision_rate": round(summary.get("collision_rate", 0.0), 4),
            # The ranking statistic: unlike `collided`, it sees fusions with
            # UNANNOTATED neighbours. See mhat.evaluation.seg_detectability.
            "over_merged": summary.get("n_over_merged", ""),
            "over_merge_rate": round(summary.get("over_merge_rate", 0.0), 4),
            "size_ratio_median": (
                round(summary["size_ratio_median"], 2)
                if summary.get("size_ratio_median") is not None else ""
            ),
            "size_ratio_p90": (
                round(summary["size_ratio_p90"], 2)
                if summary.get("size_ratio_p90") is not None else ""
            ),
            "offset_median": (
                round(summary["offset_median"], 3)
                if summary.get("offset_median") is not None else ""
            ),
            "offset_p90": (
                round(summary["offset_p90"], 3)
                if summary.get("offset_p90") is not None else ""
            ),
            "offset_over_2um": summary.get("offset_over_2um", ""),
            "recall_frag": round(summary["recall"]["frag"], 4),
            "recall_ungated": round(summary["recall"]["ungated"], 4),
            "n_markers": summary["n_markers"],
            "misses": len(summary["misses"]),
            "misses_a_filter_would_fix": summary["gated_misses_rescued_by_ungated"],
            "gained": "",
            "lost": "",
        }
        if base_detected is not None:
            here = detected_set(payload)
            row["gained"] = len(here - base_detected)
            row["lost"] = len(base_detected - here)
        for path, value in record.get("overrides", {}).items():
            row[path.split(".")[-1]] = value
        rows.append(row)

    fields = list(dict.fromkeys(key for row in rows for key in row))
    results_path = sweep_dir / "results.csv"
    with open(results_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Rank on over-merge, not collisions: collisions need two annotated markers
    # to share a candidate and so miss most fusions on sparsely annotated data.
    # Older detectability.json files have no over_merge_rate; they sort last
    # rather than tying at 0 and looking like winners.
    rows.sort(key=lambda row: (
        row["over_merge_rate"] if row.get("over_merged") != "" else float("inf"),
        row["collided"] if row["collided"] != "" else float("inf"),
        -row["recall_gated"],
    ))
    lines = [
        f"Segmentation sweep: {manifest['sweep_id']}",
        f"{manifest['experiment']} / {manifest['dataset']}",
        "",
        f"{'run':<24}{'overmrg':>9}{'srP90':>8}{'collided':>9}{'off_med':>9}"
        f"{'recall':>8}{'miss':>6}{'+':>5}{'-':>5}",
        "-" * 81,
    ]
    for row in rows:
        over = (f"{row['over_merge_rate'] * 100:8.1f}%"
                if row.get("over_merged") != "" else f"{'--':>9}")
        lines.append(
            f"  {row['label']:<22}{over}{str(row['size_ratio_p90']):>8}"
            f"{str(row['collided']):>9}{str(row['offset_median']):>9}"
            f"{row['recall_gated']:>8.4f}{row['misses']:>6}"
            f"{str(row['gained']):>5}{str(row['lost']):>5}"
        )
    lines += [
        "",
        "overmrg = THE RANKING COLUMN: detected markers whose smallest covering "
        "candidate is",
        "          bigger than one cell, i.e. fused with a neighbour. Unlike "
        "'collided' it does",
        "          not need the neighbour to be annotated, so it sees the "
        "fusions that matter on",
        "          sparsely annotated data. '--' = scored before the metric "
        "existed; re-run the",
        "          detectability step for that run. srP90 = 90th pct candidate "
        "volume / marker volume.",
        "collided = GT markers whose best candidate is another marker's too "
        "(lower is better;",
        "           these are links MaxParents(1) forbids at any cost).  "
        "off_med = median",
        "           candidate-centroid offset in um, against ~1 um of real "
        "per-frame drift.",
    ]
    if baseline_label:
        lines += ["", f"+/- are per-marker detection gains and losses against '{baseline_label}'."]
    if missing:
        lines += ["", f"No results yet for: {', '.join(missing)}"]

    report = "\n".join(lines)
    (sweep_dir / "results.txt").write_text(report + "\n")
    print(report)
    print(f"\nWrote {results_path}")


if __name__ == "__main__":
    main()
