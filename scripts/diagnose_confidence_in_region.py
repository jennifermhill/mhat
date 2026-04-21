"""Diagnose confidence distribution WITHIN segmentation regions on NC281.

Background pixels are already excluded since flow is only aggregated over segmentation
fragments. This script characterizes the confidence distribution among in-region pixels
so we can pick a meaningful threshold.
"""
import numpy as np
import zarr


def main():
    flow_3d_path = "C:/Users/hillj/Documents/mhat/experiments/opticalflow/Fluo-C3DL-MDA231/01_cells/opticalflow_3d/2026-02-17_15-55-00/flow.zarr"
    seg_path = "C:/Users/hillj/Documents/mhat/experiments/segmentation/Fluo-C3DL-MDA231/01_cells/2026-04-03_11-09-49/data.zarr"

    flow_zarr = zarr.open(flow_3d_path, mode="r")
    flow = flow_zarr["flow_raw"][:]
    conf = flow_zarr["confidence"][:]
    print(f"flow shape: {flow.shape}")
    print(f"conf shape: {conf.shape}")

    # Load fragments (segmentation)
    seg_zarr = zarr.open(seg_path, mode="r")
    # Try common paths
    frag = None
    for name in ["fov=0/channel=fragments", "fragments"]:
        try:
            frag = seg_zarr[name][:]
            print(f"Found fragments at: {name}, shape {frag.shape}")
            break
        except Exception:
            continue
    if frag is None:
        print("Couldn't find fragments array")
        return

    # fragments shape is likely (T, Z, Y, X). Align to conf's T length if needed.
    t_match = min(frag.shape[0], conf.shape[0])
    frag = frag[:t_match]
    conf = conf[:t_match]
    flow = flow[:t_match]

    # In-region mask: any nonzero fragment label
    in_region_mask = frag > 0
    print(f"Total pixels: {frag.size}")
    print(f"In-region pixels: {np.sum(in_region_mask)} ({100*np.mean(in_region_mask):.2f}%)")

    # Confidence distribution WITHIN regions
    conf_in = conf[in_region_mask]
    print()
    print("=" * 80)
    print("Confidence distribution WITHIN segmentation regions")
    print("=" * 80)
    print(
        f"mean={np.mean(conf_in):.4g}  std={np.std(conf_in):.4g}  "
        f"min={np.min(conf_in):.4g}  max={np.max(conf_in):.4g}"
    )
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{p:2d} = {np.percentile(conf_in, p):12.4g}")

    # Distribution of |conf| within regions (since sign is ambiguous for Farneback)
    abs_conf_in = np.abs(conf_in)
    print()
    print("|confidence| distribution WITHIN regions")
    print(
        f"mean={np.mean(abs_conf_in):.4g}  std={np.std(abs_conf_in):.4g}  "
        f"min={np.min(abs_conf_in):.4g}  max={np.max(abs_conf_in):.4g}"
    )
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{p:2d} = {np.percentile(abs_conf_in, p):12.4g}")

    # Flow magnitudes binned by in-region |conf|
    print()
    print("=" * 80)
    print("Flow magnitudes by in-region |conf| quantile (pixel units, unscaled)")
    print("=" * 80)
    vx = flow[..., 0][in_region_mask]
    vy = flow[..., 1][in_region_mask]
    vz = flow[..., 2][in_region_mask]

    quantile_edges = np.quantile(abs_conf_in, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    print(
        f"{'|conf| bin':<32s}  {'|vx|_mean':>10s}  {'|vy|_mean':>10s}  "
        f"{'|vz|_mean':>10s}  {'vz_std':>10s}  {'frac':>7s}"
    )
    for i in range(5):
        lo, hi = quantile_edges[i], quantile_edges[i + 1]
        if i < 4:
            mask = (abs_conf_in >= lo) & (abs_conf_in < hi)
        else:
            mask = (abs_conf_in >= lo) & (abs_conf_in <= hi)
        frac = np.mean(mask)
        if frac > 0:
            print(
                f"[{lo:12.3g}, {hi:12.3g}]  {np.mean(np.abs(vx[mask])):10.3f}  "
                f"{np.mean(np.abs(vy[mask])):10.3f}  "
                f"{np.mean(np.abs(vz[mask])):10.3f}  {np.std(vz[mask]):10.3f}  "
                f"{frac:6.3%}"
            )

    # Per-node confidence: for each labeled fragment, what fraction of its pixels
    # would pass various |conf| thresholds?
    print()
    print("=" * 80)
    print("Per-node: fraction of in-region pixels passing threshold")
    print("=" * 80)
    thresholds = [np.percentile(abs_conf_in, p) for p in [25, 50, 75, 90]]
    threshold_labels = ["p25", "p50", "p75", "p90"]
    # Sample a subset of nodes to keep runtime reasonable
    from skimage import measure

    # Work on one frame for speed
    for tp in [0, 5, 10]:
        frag_t = frag[tp]
        conf_t = np.abs(conf[tp])
        if not np.any(frag_t > 0):
            continue
        props = measure.regionprops(frag_t)
        frac_passing = {lbl: [] for lbl in threshold_labels}
        for p in props:
            pixel_confs = conf_t[p.slice][p.image]  # type: ignore
            for thr, lbl in zip(thresholds, threshold_labels):
                frac_passing[lbl].append(np.mean(pixel_confs > thr))
        print(f"\nFrame {tp}: {len(props)} fragments")
        for lbl, thr in zip(threshold_labels, thresholds):
            fracs = np.array(frac_passing[lbl])
            print(
                f"  threshold={lbl} ({thr:.3e}): mean per-fragment pass rate = "
                f"{np.mean(fracs):.3f}, median={np.median(fracs):.3f}, "
                f"fragments with <10% passing = {np.sum(fracs < 0.1)}"
            )


if __name__ == "__main__":
    main()
