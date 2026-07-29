"""Ground-truth drift diagnostics: how well does the optical flow predict motion?

For every ground truth edge this compares the raw frame-to-frame displacement
against the flow-corrected residual ``|pos_u + flow(u) - pos_v|`` — the quantity
the ILP actually pays for through ``drift_weight``/``drift_constant``. The flow
field used by the tracking run is always reported; if the sibling flow variant
(Farneback vs Lucas-Kanade) exists on disk it is reported alongside, which is
what the original ``compare_gt_drift.py`` did by hand.
"""

from pathlib import Path

import geff
import numpy as np
import zarr
from numpy import linalg

from mhat.evaluation._report import Report

FLOW_GROUP = "flow_raw"


def _open_flow(flow_dir):
    """Open ``flow.zarr/flow_raw`` under a flow result dir, or None."""
    if flow_dir is None:
        return None
    path = Path(flow_dir) / "flow.zarr"
    if not path.exists():
        return None
    try:
        return zarr.open(str(path), mode="r")[FLOW_GROUP]
    except (KeyError, FileNotFoundError, ValueError):
        return None


def flow_variants(config, run_config):
    """Locate the flow fields for this run plus any sibling variant on disk.

    Returns a list of dicts: ``{"name", "flow_3d", "flow_2d", "used_by_run"}``.
    """
    flow_result = run_config.get("flow_result")
    if not flow_result:
        return []

    flow_base = (Path(config["input_base_dir"]) / "opticalflow" /
                 config["experiment"] / config["dataset"])
    used_lk = bool(run_config.get("use_lk", False))

    variants = []

    fb_3d = _open_flow(flow_base / "opticalflow_3d" / flow_result)
    fb_2d = _open_flow(flow_base / "opticalflow_2d" / flow_result)
    if fb_3d is not None or fb_2d is not None:
        variants.append({"name": "farneback", "flow_3d": fb_3d, "flow_2d": fb_2d,
                         "used_by_run": not used_lk})

    lk_3d = _open_flow(flow_base / "opticalflow_lucaskanade" / flow_result)
    if lk_3d is not None:
        variants.append({"name": "lucaskanade", "flow_3d": lk_3d, "flow_2d": None,
                         "used_by_run": used_lk})

    return variants


def _flow_frames(variant, t):
    """The (3D, 2D) flow frames at time t; None past the end, as in run_tracking."""
    def _frame(arr):
        if arr is None or t >= arr.shape[0]:
            return None
        return np.asarray(arr[t])

    return _frame(variant["flow_3d"]), _frame(variant["flow_2d"])


def _component_sources(f3, f2, ndim):
    """Per-axis (array, trailing index) pairs, mirroring nodes_from_segmentation.

    Flow arrays store their vector components in (x, y, z) order in the trailing
    axis. In 3D the Z component comes from the 3D field and X/Y from the 2D field
    when it exists (Farneback), otherwise everything comes from one field (LK).
    """
    src = f2 if f2 is not None else f3
    if ndim == 3:
        return [(f3, 2), (src, 1), (src, 0)]
    return [(src, 1), (src, 0)]


def _point_flow(variant, t, indices, scale_spatial):
    """Flow vectors (world units, spatial-axis order) sampled at voxel ``indices``."""
    ndim = indices.shape[1]
    out = np.zeros((len(indices), ndim), dtype=float)

    f3, f2 = _flow_frames(variant, t)
    if f3 is None and f2 is None:
        return out

    idx = tuple(indices[:, d] for d in range(ndim))
    for axis, (arr, comp) in enumerate(_component_sources(f3, f2, ndim)):
        if arr is not None:
            out[:, axis] = arr[idx][:, comp] * scale_spatial[axis]
    return out


def _region_flow(variant, t, seg_frame, labels, scale_spatial):
    """Flow vectors averaged over each GT label's region, as the tracker does.

    ``mhat.tracking.utils.nodes_from_segmentation`` averages the flow over the
    whole segment rather than sampling one voxel, which matters a lot for
    Lucas-Kanade (its per-voxel outliers reach hundreds of pixels). Labels absent
    from the frame come back as NaN so the caller can fall back to point sampling.
    """
    from scipy.ndimage import mean as ndi_mean

    ndim = seg_frame.ndim
    out = np.full((len(labels), ndim), np.nan, dtype=float)

    f3, f2 = _flow_frames(variant, t)
    if f3 is None and f2 is None:
        out[:] = 0.0
        return out

    present = np.unique(seg_frame)
    present = present[present != 0]
    if present.size == 0:
        return out

    index_of = {int(lab): i for i, lab in enumerate(present)}
    for axis, (arr, comp) in enumerate(_component_sources(f3, f2, ndim)):
        if arr is None:
            out[:, axis] = 0.0
            continue
        means = np.asarray(ndi_mean(arr[..., comp], labels=seg_frame, index=present),
                           dtype=float) * scale_spatial[axis]
        for i, lab in enumerate(labels):
            j = index_of.get(int(lab))
            if j is not None:
                out[i, axis] = means[j]
    return out


def gt_label_map(gt_graph, gt_metadata):
    """{node: segmentation label} for the GT graph, or None if unavailable."""
    label_prop = "track_id"
    related = getattr(gt_metadata, "related_objects", None)
    if related:
        for ro in related:
            if ro.type == "labels" and ro.label_prop:
                label_prop = ro.label_prop
                break

    label_map = {}
    for node, data in gt_graph.nodes(data=True):
        if label_prop not in data:
            return None
        label_map[node] = int(data[label_prop])
    return label_map


def compare_gt_drift(config, gt_data_dir, pred_data_dir, run_config, output_dir=None,
                     plots=True, echo=True):
    """Compare raw vs flow-corrected displacement over all GT edges.

    Args:
        config (dict): Evaluation config (``input_base_dir``/``experiment``/``dataset``).
        gt_data_dir (Path): Directory holding ``correct_tracks.zarr``.
        pred_data_dir (Path): Tracking result directory (used for the axis scale).
        run_config (dict): Tracking config of the run (``flow_result``, ``use_lk``).
        output_dir (Path | None): Where to write ``gt_drift_report.txt`` and the
            histogram. Nothing is written if None.
        plots (bool): Write the histogram PNG.
        echo (bool): Print the report to stdout as it is built.

    Returns:
        dict: Displacement stats per flow variant.
    """
    gt_data_dir = Path(gt_data_dir)
    report = Report("GT drift analysis (raw vs flow-corrected displacement)", echo=echo)

    gt_graph, gt_metadata = geff.read(gt_data_dir / "correct_tracks.zarr")
    axes = gt_metadata.axes or []
    names = [a.name for a in axes if getattr(a, "type", None) != "time"] or ["z", "y", "x"]
    scale = [a.scale if a.scale is not None else 1.0 for a in axes] or [1.0] * (len(names) + 1)
    scale_spatial = np.array(scale[-len(names):], dtype=float)

    edges = list(gt_graph.edges())
    if not edges:
        report.line("GT graph has no edges - nothing to compare.")
        if output_dir is not None:
            report.write(Path(output_dir) / "gt_drift_report.txt")
        return {"n_gt_edges": 0, "variants": {}}

    # Positions are stored in world units; recover voxel indices for flow lookup.
    by_time = {}
    for u, v in edges:
        nu, nv = gt_graph.nodes[u], gt_graph.nodes[v]
        t = int(nu["time"])
        pos_u = np.array([float(nu[n]) for n in names])
        pos_v = np.array([float(nv[n]) for n in names])
        by_time.setdefault(t, []).append((u, pos_u, pos_v))

    no_flow = np.concatenate([
        linalg.norm(np.array([e[2] - e[1] for e in entries]), axis=1)
        for entries in by_time.values()
    ])

    report.line("")
    report.line(f"GT edges: {len(edges)}   spatial axes: {names}   "
                f"scale: {[round(float(s), 4) for s in scale_spatial]}")

    # Prefer averaging the flow over each GT segment, which is what
    # nodes_from_segmentation does; fall back to sampling at the centroid.
    gt_seg = None
    label_map = None
    for candidate in ("correct_seg.zarr", "segmentation"):
        seg_path = gt_data_dir / candidate
        if seg_path.exists():
            try:
                gt_seg = zarr.open(str(seg_path), mode="r")
            except (KeyError, FileNotFoundError, ValueError):
                gt_seg = None
            break
    if gt_seg is not None and gt_seg.ndim - 1 == len(names):
        label_map = gt_label_map(gt_graph, gt_metadata)
    use_regions = gt_seg is not None and label_map is not None
    report.line("Flow sampling: " + ("mean over the GT segment (matches tracking)"
                                     if use_regions else
                                     "single voxel at the GT centroid (no GT segmentation "
                                     "available; outlier-sensitive for Lucas-Kanade)"))

    variants = flow_variants(config, run_config)
    if not variants:
        report.line("No optical flow results found for this run "
                    "(no flow_result in the tracking config, or the flow dirs are missing).")

    results = {"none": {"dist": no_flow, "flow_mag": None, "used_by_run": not variants}}

    for variant in variants:
        dists = []
        mags = []
        n_fallback = 0
        for t, entries in by_time.items():
            nodes = [e[0] for e in entries]
            starts = np.array([e[1] for e in entries])
            ends = np.array([e[2] for e in entries])
            arr = variant["flow_3d"] if variant["flow_3d"] is not None else variant["flow_2d"]
            shape = arr.shape[1:1 + len(names)]
            indices = np.rint(starts / scale_spatial).astype(int)
            for d in range(len(names)):
                indices[:, d] = np.clip(indices[:, d], 0, shape[d] - 1)

            vecs = _point_flow(variant, t, indices, scale_spatial)
            if use_regions and t < gt_seg.shape[0]:
                region = _region_flow(variant, t, np.asarray(gt_seg[t]),
                                      [label_map[n] for n in nodes], scale_spatial)
                missing = np.isnan(region).any(axis=1)
                n_fallback += int(missing.sum())
                vecs = np.where(missing[:, None], vecs, region)

            dists.append(linalg.norm(starts + vecs - ends, axis=1))
            mags.append(linalg.norm(vecs, axis=1))
        if n_fallback:
            report.line(f"  {variant['name']}: {n_fallback} GT nodes had no label in the "
                        "GT segmentation; centroid-sampled instead.")
        results[variant["name"]] = {
            "dist": np.concatenate(dists),
            "flow_mag": np.concatenate(mags),
            "used_by_run": variant["used_by_run"],
        }

    report.line("")
    report.line("=" * 80)
    report.line(f"{'Flow':<16} {'Used':<6} {'Mean':>10} {'Std':>10} {'Median':>10} "
                f"{'Flow mag':>12}")
    report.line("-" * 80)
    summary = {}
    for name, res in results.items():
        dist = res["dist"]
        mag = res["flow_mag"]
        mag_str = f"{mag.mean():.3f}" if mag is not None else "-"
        report.line(f"{name:<16} {'yes' if res['used_by_run'] else '':<6} "
                    f"{dist.mean():>10.3f} {dist.std():>10.3f} {np.median(dist):>10.3f} "
                    f"{mag_str:>12}")
        summary[name] = {
            "mean": float(dist.mean()),
            "std": float(dist.std()),
            "median": float(np.median(dist)),
            "used_by_run": bool(res["used_by_run"]),
        }
        if mag is not None:
            summary[name]["flow_mag_mean"] = float(mag.mean())
            summary[name]["flow_mag_std"] = float(mag.std())
    report.line("=" * 80)
    report.line("Mean/std/median of |pos_u + flow(u) - pos_v| over all GT edges; "
                "'none' is the raw displacement.")
    report.line("This is the drift_dist the ILP would see for a perfect linking, so it "
                "bounds a sensible drift_weight/drift_constant break-even point.")

    if output_dir is not None:
        output_dir = Path(output_dir)
        if plots:
            plot_name = _plot_drift(results, output_dir)
            if plot_name:
                report.line("")
                report.line(f"Histogram saved to {output_dir / plot_name}")
        report.write(output_dir / "gt_drift_report.txt")

    return {"n_gt_edges": len(edges), "variants": summary}


def _plot_drift(results, output_dir):
    """Overlaid histogram of GT-edge displacement per flow variant."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_vals = np.concatenate([r["dist"] for r in results.values()])
    if all_vals.size == 0:
        return None
    bins = np.histogram_bin_edges(all_vals, bins=40)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, res in results.items():
        label = f"{name} (mean={res['dist'].mean():.2f})"
        if res["used_by_run"]:
            label += " [used]"
        ax.hist(res["dist"], bins=bins, histtype="step", linewidth=1.8, label=label)
    ax.set_xlabel("GT edge displacement (world units)")
    ax.set_ylabel("# of GT edges")
    ax.set_title("GT edge drift: raw vs flow-corrected")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "hist_gt_drift.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name
