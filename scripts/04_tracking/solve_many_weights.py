"""Solve many weight settings against one candidate graph.

Every config in a GT-amount sweep points at the same test dataset, so running
run_tracking.py once per config would rebuild the identical candidate graph
dozens of times (~4 min each) to do ~2 s of actual solving. This builds the
graph once and loops.

Solving does not mutate the graph and each iteration gets a fresh motile.Solver,
so results are identical to the per-config path — verify that on a couple of
tokens before trusting the batch.

Usage:
    python scripts/04_tracking/solve_many_weights.py \
        configs/tracking/NC281-sparse-label/02_nuclei_denoised_test/gt_amount/*.toml
    python scripts/04_tracking/solve_many_weights.py --config-dir <dir> [--only TOKEN ...]
"""

from __future__ import annotations

import argparse
import glob
import traceback
from pathlib import Path

import geff
import numpy as np
import toml
import zarr

from mhat.tracking import utils
from mhat.tracking.pipeline import build_track_graph, resolve_input_dirs
from mhat.tracking.solve_with_motile import solve_with_motile
from motile_toolbox.visualization.napari_utils import assign_tracklet_ids


def _graph_key(config) -> tuple:
    """Configs sharing this key share a candidate graph."""
    return (
        config["experiment"],
        config["dataset"],
        config["seg_result"],
        config.get("flow_result"),
        config.get("use_lk", False),
        config["size_threshold"],
        config["max_edge_distance"],
        config["max_children"],
        config["divisions"],
        config["merges"],
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

    keys = {_graph_key(c) for _, c in configs}
    if len(keys) != 1:
        parser.error(
            f"configs describe {len(keys)} different candidate graphs; "
            "split them into one batch per graph"
        )

    first = configs[0][1]
    raw_dir, seg_dir, flow_dirs, _gt = resolve_input_dirs(first)
    output_base_dir = Path(first["output_base_dir"])
    dataset_out = output_base_dir / "tracking" / first["experiment"] / first["dataset"]

    print(f"Building the shared candidate graph once for {len(configs)} configs...")
    track_graph, fragments, merge_history, exclusion_sets, scale, axes = build_track_graph(
        first, raw_dir, seg_dir, flow_dirs
    )
    no_merges = len(merge_history) == 0
    print(f"  {len(track_graph.nodes)} nodes, {len(track_graph.edges)} edges")

    # build_track_graph returns fragments as a lazy zarr. Collect the leaf ids one
    # frame at a time, once — they are the same for every config below.
    frag_ids = set()
    for t in range(fragments.shape[0]):
        frag_ids.update(int(v) for v in np.unique(fragments[t]))
    frag_ids.discard(0)
    max_frag_id = max(frag_ids) if frag_ids else 0

    failures = []
    for path, config in configs:
        exp_uid = config.get("exp_uid") or path.stem
        out_dir = dataset_out / exp_uid
        if args.skip_existing and (out_dir / "pred_tracks.zarr").is_dir():
            print(f"{exp_uid}: already present, skipping")
            continue
        print(f"\n=== {exp_uid} ===")
        out_dir.mkdir(parents=True, exist_ok=True)
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

            lookup = utils.get_solution_lookup(
                merge_history, solution_graph, frag_ids, max_frag_id, fragments.dtype
            )
            assign_tracklet_ids(solution_graph)

            seg_root = zarr.open(
                out_dir / "pred_seg.zarr",
                mode="w",
                shape=fragments.shape,
                chunks=(1, 1, 512, 512),
                dtype=np.uint32,
            )
            if axes is not None:
                seg_root.attrs["axes"] = axes
            for t in range(fragments.shape[0]):
                seg_root[t] = lookup[fragments[t]]

            metadata = geff.GeffMetadata(
                directed=True,
                related_objects=[
                    {"type": "labels", "path": "../pred_seg.zarr", "label_prop": "label"}
                ],
                node_props_metadata={},
                edge_props_metadata={},
            )
            geff.write(
                solution_graph,
                out_dir / "pred_tracks.zarr",
                axis_names=["time", "z", "y", "x"],
                axis_types=["time", "space", "space", "space"],
                axis_scales=scale,
                metadata=metadata,
                overwrite=True,
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
