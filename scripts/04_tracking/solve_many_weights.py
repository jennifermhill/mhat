"""Solve many weight settings against one candidate graph.

Every config in a GT-amount sweep points at the same test dataset, so running
run_tracking.py once per config would rebuild the identical candidate graph
dozens of times (~4 min each) to do ~2 s of actual solving. This builds the
graph once and loops.

Solving does not mutate the graph and each iteration gets a fresh motile.Solver,
so results are identical to the per-config path — verify that on a couple of
tokens before trusting the batch.

Each config's run is a tracking run at ``<output_dir>/tracking/<exp_uid>``
(``exp_uid`` defaults to the config's file stem; a grouped uid such as
``sweeps/<id>/runs/<token>`` is fine). An existing run is never redone: with
``--skip-existing`` a finished one is skipped, otherwise it is reported.

Usage:
    python scripts/04_tracking/solve_many_weights.py <dir>/*.toml
    python scripts/04_tracking/solve_many_weights.py --config-dir <dir> [--only TOKEN ...]
"""

from __future__ import annotations

import argparse
import glob
import traceback
from pathlib import Path

import toml

from mhat.tracking.pipeline import (
    build_track_graph,
    resolve_inputs,
    write_tracking_outputs,
)
from mhat.tracking.solve_with_motile import solve_with_motile


def _graph_key(config) -> tuple:
    """Configs sharing this key share a candidate graph."""
    return (
        config["output_dir"],
        config["seg_result"],
        config.get("flow_result"),
        config["size_threshold"],
        config["max_edge_distance"],
        config["max_children"],
        config["divisions"],
        config.get("min_merge_cost", config.get("min_merge_score", 0.0)),
        config.get("max_merge_cost", config.get("max_merge_score", 1.0)),
        config.get("z_flow_conf_threshold"),
        config.get("z_flow_min_pass_pixels", 10),
        tuple(config.get("drift_distance", ())),
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
    # run_tracking.py z-scores the cost parameters for normalize_features; this
    # loop calls the solver directly and does not, so it would silently solve
    # with the raw weights. Refuse until the two share their post-graph path.
    zscored = [p.name for p, c in configs if c.get("normalize_features") is not None]
    if zscored:
        parser.error(
            f"normalize_features is not supported here (use run_tracking.py): {zscored}"
        )

    keys = {_graph_key(c) for _, c in configs}
    if len(keys) != 1:
        parser.error(
            f"configs describe {len(keys)} different candidate graphs; "
            "split them into one batch per graph"
        )

    first = configs[0][1]
    ds, raw_dir, seg_dir, flow_dirs, _gt_tracks, _gt_seg = resolve_inputs(first)
    ds.bind_raw()

    print(f"Building the shared candidate graph once for {len(configs)} configs...")
    track_graph, fragments, merge_history, exclusion_sets, scale, axes = build_track_graph(
        first, raw_dir, seg_dir, flow_dirs
    )
    no_merges = len(merge_history) == 0
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
            solution_graph = solve_with_motile(
                config, track_graph, exclusion_sets, no_merges=no_merges
            )
            if solution_graph is None or solution_graph.number_of_nodes() == 0:
                print(f"  {exp_uid}: EMPTY solution; no outputs written")
                failures.append((exp_uid, "empty_solution"))
                continue

            # Same writer as run_tracking.py / fit_weights_ssvm.py, so the outputs
            # are byte-compatible with the per-config path at either rank.
            write_tracking_outputs(
                solution_graph, fragments, merge_history, scale, axes, out_dir
            )
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
