"""Diagnose reliability of 3D optical flow on NC281.

Reports per-axis flow statistics for flow_3d and flow_2d, and compares them
by voxel-scaling to physical units to see if Z flow magnitude is plausible
vs actual Z motion of cells.
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
    parser.add_argument(
        "--flow_2d",
        default="C:/Users/hillj/Documents/mhat/experiments/opticalflow/NC281-Fl2mSiH2B/03_nuclei/opticalflow_2d/2026-03-24_17-16-56/flow.zarr",
    )
    parser.add_argument("--scale_z", type=float, default=2.110045248868777)
    parser.add_argument("--scale_y", type=float, default=0.65)
    parser.add_argument("--scale_x", type=float, default=0.65)
    args = parser.parse_args()

    print(f"Loading 3D flow from {args.flow_3d}")
    f3 = zarr.open(args.flow_3d, mode="r")
    # Try to find the flow array
    for name in ["flow_raw", "flow", "fov=0/channel=flow_raw", "fov=0/channel=flow"]:
        try:
            arr = f3[name]
            print(f"Found 3D flow at group: {name}, shape={arr.shape}, dtype={arr.dtype}")
            flow_3d_data = arr[:]
            break
        except Exception:
            continue
    else:
        # fallback: traverse
        print("Traversing zarr tree:")
        print(f3.tree())
        return

    print(f"Loading 2D flow from {args.flow_2d}")
    f2 = zarr.open(args.flow_2d, mode="r")
    for name in ["flow_raw", "flow", "fov=0/channel=flow_raw", "fov=0/channel=flow"]:
        try:
            arr = f2[name]
            print(f"Found 2D flow at group: {name}, shape={arr.shape}, dtype={arr.dtype}")
            flow_2d_data = arr[:]
            break
        except Exception:
            continue
    else:
        print("Traversing zarr tree:")
        print(f2.tree())
        return

    # flow_3d shape (T, Z, Y, X, 3) with channels (vx, vy, vz)
    # flow_2d shape (T, Z, Y, X, 2) with channels (vx, vy)
    print()
    print("=" * 80)
    print("3D FLOW STATISTICS (pixel units, raw from zarr)")
    print("=" * 80)
    print(f"Shape: {flow_3d_data.shape}")
    if flow_3d_data.shape[-1] >= 3:
        vx = flow_3d_data[..., 0]
        vy = flow_3d_data[..., 1]
        vz = flow_3d_data[..., 2]
        for name, arr, scale in [
            ("vx (pixel)", vx, 1.0),
            ("vy (pixel)", vy, 1.0),
            ("vz (pixel)", vz, 1.0),
            ("vx (micron)", vx, args.scale_x),
            ("vy (micron)", vy, args.scale_y),
            ("vz (micron)", vz, args.scale_z),
        ]:
            a = arr * scale
            print(
                f"{name:18s}  mean={np.mean(a):8.3f}  std={np.std(a):8.3f}  "
                f"min={np.min(a):8.3f}  max={np.max(a):8.3f}  "
                f"|mean|={np.mean(np.abs(a)):8.3f}  p95={np.percentile(np.abs(a), 95):8.3f}"
            )
    print()
    print("=" * 80)
    print("2D FLOW STATISTICS (pixel units, raw from zarr)")
    print("=" * 80)
    print(f"Shape: {flow_2d_data.shape}")
    if flow_2d_data.shape[-1] >= 2:
        vx2 = flow_2d_data[..., 0]
        vy2 = flow_2d_data[..., 1]
        for name, arr, scale in [
            ("vx_2d (pixel)", vx2, 1.0),
            ("vy_2d (pixel)", vy2, 1.0),
            ("vx_2d (micron)", vx2, args.scale_x),
            ("vy_2d (micron)", vy2, args.scale_y),
        ]:
            a = arr * scale
            print(
                f"{name:18s}  mean={np.mean(a):8.3f}  std={np.std(a):8.3f}  "
                f"min={np.min(a):8.3f}  max={np.max(a):8.3f}  "
                f"|mean|={np.mean(np.abs(a)):8.3f}  p95={np.percentile(np.abs(a), 95):8.3f}"
            )

    # Compare 3D vs 2D for XY axes: they should roughly agree if both flows are sensible
    if flow_3d_data.shape[-1] >= 3 and flow_2d_data.shape[-1] >= 2:
        print()
        print("=" * 80)
        print("3D vs 2D comparison (XY axes, pixel units)")
        print("=" * 80)
        vx = flow_3d_data[..., 0]
        vy = flow_3d_data[..., 1]
        # handle possible T dimension differences — take min
        t_min = min(vx.shape[0], vx2.shape[0])
        vx = vx[:t_min]
        vy = vy[:t_min]
        vx2 = vx2[:t_min]
        vy2 = vy2[:t_min]
        # Correlation
        vx_corr = np.corrcoef(vx.ravel(), vx2.ravel())[0, 1]
        vy_corr = np.corrcoef(vy.ravel(), vy2.ravel())[0, 1]
        print(f"Pearson corr(3D vx, 2D vx) = {vx_corr:.4f}")
        print(f"Pearson corr(3D vy, 2D vy) = {vy_corr:.4f}")
        print(
            "Interpretation: if correlations are high (>0.7), 3D flow is consistent in XY. "
            "If Z flow magnitude is very different from XY flow magnitude (esp. |vz|_mean much "
            "smaller than |vx|/|vy|), Z flow is likely noise or poorly constrained."
        )


if __name__ == "__main__":
    main()
