"""Solve many weight settings against one candidate graph.

Every config in a GT-amount sweep points at the same test dataset, so running
run_tracking.py once per config would rebuild the identical candidate graph
dozens of times (~4 min each) to do ~2 s of actual solving. This builds the
graph once and loops.

The graph is built, solved and written by run_tracking.py's own functions
(``build_candidate_graph``, ``solve_candidate_graph``, ``write_solution``), and
configs are grouped by its graph-cache key, so each run here is the run
run_tracking.py would make from the same config: same outputs, including
``normalized_merge_history.csv``, ``candidate_edges.npy`` and
``normalize_features``.

Each config's run is a tracking run at ``<output_dir>/tracking/<exp_uid>``
(``exp_uid`` defaults to the config's file stem; a grouped uid such as
``sweeps/<id>/runs/<token>`` is fine). An existing run is never redone: with
``--skip-existing`` a finished one is skipped, otherwise it is reported. An
empty solution writes no ``pred_tracks.zarr``, which is how
``collect_gt_amount.py`` recognizes it.

Usage:
    python scripts/04_tracking/solve_many_weights.py <dir>/*.toml
    python scripts/04_tracking/solve_many_weights.py --config-dir <dir> [--only TOKEN ...]
"""

from __future__ import annotations

import argparse
import glob
import json
import traceback
from pathlib import Path

import toml
from run_tracking import (
    build_candidate_graph,
    graph_cache_key,
    input_fingerprint,
    solve_candidate_graph,
    uses_drift_distance,
    write_solution,
)

from mhat.tracking.pipeline import resolve_inputs


def graph_identity(config) -> str:
    """Configs with equal identities share a candidate graph.

    The graph-cache key (run_tracking.py), so the two can never disagree about
    which parameters shape the graph, plus the output directory, since uids
    only name runs within one.
    """
    ds, _raw_dir, seg_dir, flow_dirs, _gt_tracks, _gt_seg = resolve_inputs(config)
    key = graph_cache_key(
        config, uses_drift_distance(flow_dirs), input_fingerprint(seg_dir, flow_dirs)
    )
    return json.dumps(
        {"output_dir": str(ds.output_dir), **key}, sort_keys=True, default=str
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="*", type=Path)
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    paths = list(args.configs)
    if args.config_dir:
        paths += [Path(p) for p in sorted(glob.glob(str(args.config_dir / "*.toml")))]
    if args.only:
        keep = set(args.only)
        paths = [p for p in paths if p.stem in keep]
    if not paths:
        parser.error("no configs given")

    configs = [(p, toml.load(p)) for p in sorted(set(paths))]
    identities = {graph_identity(c) for _, c in configs}
    if len(identities) != 1:
        parser.error(
            f"configs describe {len(identities)} different candidate graphs; "
            "split them into one batch per graph"
        )

    first = configs[0][1]
    ds, raw_dir, seg_dir, flow_dirs, _gt_tracks, _gt_seg = resolve_inputs(first)
    for _, config in configs:
        # Each config names its own raw_path; all must match the directory's.
        resolve_inputs(config)[0].bind_raw()

    print(f"Building the shared candidate graph once for {len(configs)} configs...")
    bundle = build_candidate_graph(first, raw_dir, seg_dir, flow_dirs)
    track_graph = bundle["track_graph"]
    print(f"  {len(track_graph.nodes)} nodes, {len(track_graph.edges)} edges")

    failures = []
    for path, config in configs:
        exp_uid = config.get("exp_uid") or path.stem
        config["exp_uid"] = exp_uid
        if args.skip_existing and (ds.track_dir(exp_uid) / "pred_tracks.zarr").is_dir():
            print(f"{exp_uid}: already present, skipping")
            continue
        print(f"\n=== {exp_uid} ===")
        try:
            out_dir = ds.prepare_run_dir("tracking", exp_uid)
        except FileExistsError as error:
            print(f"  {error}")
            failures.append((exp_uid, "run directory exists"))
            continue
        with open(out_dir / "config.toml", "w") as f:
            toml.dump(config, f)
        try:
            solution_graph = solve_candidate_graph(config, bundle, out_dir)
            if solution_graph is None or solution_graph.number_of_nodes() == 0:
                print(f"  {exp_uid}: EMPTY solution; no outputs written")
                failures.append((exp_uid, "empty_solution"))
                continue
            write_solution(solution_graph, bundle, seg_dir, out_dir)
            print(
                f"  {exp_uid}: {solution_graph.number_of_nodes()} nodes, "
                f"{solution_graph.number_of_edges()} edges -> {out_dir}"
            )
        except Exception:
            print(f"  {exp_uid} FAILED:\n{traceback.format_exc()}")
            failures.append((exp_uid, "exception"))

    if failures:
        print(f"\n{len(failures)} run(s) produced no output:")
        for exp_uid, why in failures:
            print(f"  {exp_uid}: {why}")


if __name__ == "__main__":
    main()
