"""Compare flow-corrected drift distances for GT edges using Farneback vs LK flow."""
import geff
import zarr
import numpy as np

# Load GT tracks
g, m = geff.read("C:/Users/hillj/Documents/mhat/experiments/tracking/NC281-Fl2mSiH2B/03_nuclei/correct_tracks.zarr")
scale = [a.scale for a in m.axes]
print(f"Scale: {scale}")

# Load flow fields
fb3d = zarr.open("C:/Users/hillj/Documents/mhat/experiments/opticalflow/NC281-Fl2mSiH2B/03_nuclei/opticalflow_3d/2026-03-24_17-16-56/flow.zarr")["flow_raw"][:]
fb2d = zarr.open("C:/Users/hillj/Documents/mhat/experiments/opticalflow/NC281-Fl2mSiH2B/03_nuclei/opticalflow_2d/2026-03-24_17-16-56/flow.zarr")["flow_raw"][:]
lk = zarr.open("C:/Users/hillj/Documents/mhat/experiments/opticalflow/NC281-Fl2mSiH2B/03_nuclei/opticalflow_lucaskanade/2026-03-24_17-16-56/flow.zarr")["flow_raw"][:]
print(f"FB3D shape: {fb3d.shape}, FB2D shape: {fb2d.shape}, LK shape: {lk.shape}")

no_flow_dists = []
fb_dists = []
lk_dists = []

for u, v in g.edges():
    nu = g.nodes[u]
    nv = g.nodes[v]
    t = int(nu["time"])

    # Scaled positions (same as tracking code)
    pos_u = np.array([nu["z"], nu["y"], nu["x"]])
    pos_v = np.array([nv["z"], nv["y"], nv["x"]])

    # Convert back to pixel coords for flow lookup
    pz = int(round(nu["z"] / scale[1]))
    py = int(round(nu["y"] / scale[2]))
    px = int(round(nu["x"] / scale[3]))
    pz = np.clip(pz, 0, fb3d.shape[1] - 1)
    py = np.clip(py, 0, fb3d.shape[2] - 1)
    px = np.clip(px, 0, fb3d.shape[3] - 1)

    # Farneback: z from 3D, y/x from 2D (matching nodes_from_segmentation logic)
    t_fb = min(t, fb3d.shape[0] - 1)
    fb3d_flow = fb3d[t_fb, pz, py, px]
    fb2d_flow = fb2d[min(t, fb2d.shape[0] - 1), pz, py, px]
    fb_flow_vec = np.array([fb3d_flow[2], fb2d_flow[1], fb2d_flow[0]])

    # LK: all from 3D
    t_lk = min(t, lk.shape[0] - 1)
    lk_flow_raw = lk[t_lk, pz, py, px]
    lk_flow_vec = np.array([lk_flow_raw[2], lk_flow_raw[1], lk_flow_raw[0]])

    no_flow_dists.append(np.linalg.norm(pos_u - pos_v))
    fb_dists.append(np.linalg.norm(pos_u + fb_flow_vec - pos_v))
    lk_dists.append(np.linalg.norm(pos_u + lk_flow_vec - pos_v))

nf = np.array(no_flow_dists)
fb = np.array(fb_dists)
lk2 = np.array(lk_dists)

print(f"\nGT edge drift distances ({len(nf)} edges):")
print(f"  No flow:   mean={nf.mean():.3f}, std={nf.std():.3f}, median={np.median(nf):.3f}")
print(f"  Farneback: mean={fb.mean():.3f}, std={fb.std():.3f}, median={np.median(fb):.3f}")
print(f"  LK:        mean={lk2.mean():.3f}, std={lk2.std():.3f}, median={np.median(lk2):.3f}")

# Also print flow magnitudes at GT positions
fb_mags = []
lk_mags = []
for u, v in g.edges():
    nu = g.nodes[u]
    t = int(nu["time"])
    pz = int(round(nu["z"] / scale[1]))
    py = int(round(nu["y"] / scale[2]))
    px = int(round(nu["x"] / scale[3]))
    pz = np.clip(pz, 0, fb3d.shape[1] - 1)
    py = np.clip(py, 0, fb3d.shape[2] - 1)
    px = np.clip(px, 0, fb3d.shape[3] - 1)
    t_fb = min(t, fb3d.shape[0] - 1)
    fb3d_flow = fb3d[t_fb, pz, py, px]
    fb2d_flow = fb2d[min(t, fb2d.shape[0] - 1), pz, py, px]
    fb_vec = np.array([fb3d_flow[2], fb2d_flow[1], fb2d_flow[0]])
    lk_vec = lk[min(t, lk.shape[0] - 1), pz, py, px][[2, 1, 0]]
    fb_mags.append(np.linalg.norm(fb_vec))
    lk_mags.append(np.linalg.norm(lk_vec))

fb_m = np.array(fb_mags)
lk_m = np.array(lk_mags)
print(f"\nFlow magnitudes at GT node positions:")
print(f"  Farneback: mean={fb_m.mean():.3f}, std={fb_m.std():.3f}")
print(f"  LK:        mean={lk_m.mean():.3f}, std={lk_m.std():.3f}")
