"""Diagnose the confidence distribution for 3D optical flow on NC281.

Reports confidence statistics to inform a confidence-based thresholding strategy
for selectively including Z flow.
"""
import argparse
import numpy as np
import zarr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flow_3d",
        default="C:/Users/hillj/Documents/mhat/experiments/opticalflow/NC281-Fl2mSiH2B/03_nuclei/opticalflow_lucaskanade/2026-03-24_17-16-56/flow.zarr",
    )
    args = parser.parse_args()

    z = zarr.open(args.flow_3d, mode="r")
    conf = z["confidence"][:]
    flow = z["flow_raw"][:]
    print(f"confidence shape: {conf.shape}, dtype: {conf.dtype}")
    print(f"flow_raw shape: {flow.shape}, dtype: {flow.dtype}")

    print()
    print("=" * 80)
    print("Confidence distribution (all pixels)")
    print("=" * 80)
    print(
        f"mean={np.mean(conf):.4g}  std={np.std(conf):.4g}  "
        f"min={np.min(conf):.4g}  max={np.max(conf):.4g}"
    )
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        print(f"  p{p:2d} = {np.percentile(conf, p):10.4g}")

    # Fraction of pixels passing various thresholds
    print()
    print("Fraction of pixels above thresholds:")
    for thr in [
        np.percentile(conf, 10),
        np.percentile(conf, 25),
        np.percentile(conf, 50),
        np.percentile(conf, 75),
        np.percentile(conf, 90),
    ]:
        frac = np.mean(conf > thr)
        print(f"  conf > {thr:10.4g}: {frac:6.3%}")

    # Relationship: does high confidence correspond to meaningful flow?
    # Bin pixels by confidence and look at |vz| in each bin
    print()
    print("=" * 80)
    print("Flow magnitudes by confidence quantile (pixel units, unscaled)")
    print("=" * 80)
    if flow.shape[-1] >= 3:
        vx = flow[..., 0]
        vy = flow[..., 1]
        vz = flow[..., 2]
        # Ensure shape matches conf
        if conf.shape == vz.shape:
            quantile_edges = np.quantile(conf, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
            print(
                f"{'Conf bin':<30s}  {'|vx|_mean':>10s}  {'|vy|_mean':>10s}  "
                f"{'|vz|_mean':>10s}  {'|vz|_std':>10s}  {'frac':>7s}"
            )
            for i in range(5):
                lo, hi = quantile_edges[i], quantile_edges[i + 1]
                if i < 4:
                    mask = (conf >= lo) & (conf < hi)
                else:
                    mask = (conf >= lo) & (conf <= hi)
                frac = np.mean(mask)
                if frac > 0:
                    vxb = np.mean(np.abs(vx[mask]))
                    vyb = np.mean(np.abs(vy[mask]))
                    vzb_mean = np.mean(np.abs(vz[mask]))
                    vzb_std = np.std(vz[mask])
                    print(
                        f"[{lo:10.3g}, {hi:10.3g}]  {vxb:10.3f}  {vyb:10.3f}  "
                        f"{vzb_mean:10.3f}  {vzb_std:10.3f}  {frac:6.3%}"
                    )
        else:
            print(f"conf shape {conf.shape} differs from vz shape {vz.shape}; skipping binned stats")

    # Look at variability of confidence within a typical cell region.
    # If confidence is spatially correlated (entire regions high or low),
    # pixel-level thresholding is more meaningful.
    # Here we just report the mean and std of confidence across voxels;
    # stronger spatial structure would show up in a more detailed analysis.


if __name__ == "__main__":
    main()
