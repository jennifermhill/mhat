from pathlib import Path
import napari
import numpy as np
import zarr
import dask.array as da
from mhat.opticalflow.visualization import generate_flow_frames

def main(path_to_raw: Path, flow_exp: str, compute: bool = False):
    raw_img = da.from_zarr(path_to_raw / '0' / '0')
    raw_img = raw_img[:50, 3, 32:132, 530:920, 400:750]
    print(f"raw img shape: {raw_img.shape}")

    flow_exp_path = path_to_raw.parent / 'opticalflow_3d' / flow_exp / 'flow.zarr'
    if not flow_exp_path.exists():
        raise FileNotFoundError(f"Flow experiment path does not exist: {flow_exp_path}")
    flow_zarr = zarr.open(flow_exp_path, mode='a')
    flow_raw = da.from_zarr(flow_exp_path / 'flow_raw')
    confidence = da.from_zarr(flow_exp_path / 'confidence')
    T, Z, Y, X, _ = flow_raw.shape
    print(f"flow raw shape: {flow_raw.shape}")
    
    flow_frames_xy_path = flow_exp_path / 'flow_frames_XY'
    if not flow_frames_xy_path.exists():
        print(f"Flow frames XY path does not exist. Creating at: {flow_frames_xy_path}")
        generate_flow_frames(flow_zarr, scale_factor=0.1, color_wheel=True)
    flow_frames_xy = da.from_zarr(flow_frames_xy_path)
    print(f"flow frames XY shape: {flow_frames_xy.shape}")

    flow_frames_z_path = flow_exp_path / 'flow_frames_Z'
    if not flow_frames_z_path.exists():
        print(f"Flow frames Z path does not exist. Creating at: {flow_frames_z_path}")
        flow_zarr.create_dataset('flow_frames_Z', shape=(T, Z, Y, X), chunks=(1, Z, Y, X), dtype=np.float32)
        flow_zarr['flow_frames_Z'][:] = flow_raw[..., 2]
    flow_frames_z = da.from_zarr(flow_frames_z_path)
    print(f"flow frames Z shape: {flow_frames_z.shape}")

    if compute:
        print("Computing frames from dask arrays...")
        raw_img = raw_img.compute()
        flow_raw = flow_raw.compute()
        flow_frames_xy = flow_frames_xy.compute()
        flow_frames_z = flow_frames_z.compute()
        confidence = confidence.compute()
        print("Computation complete.")

    viewer = napari.Viewer()
    viewer.add_image(raw_img, name='raw')
    viewer.add_image(flow_raw, name='flow raw')
    viewer.add_image(flow_frames_xy, name='flow XY', blending='additive')
    viewer.add_image(flow_frames_z, name='flow Z',  colormap='berlin', blending='additive')
    viewer.add_image(confidence, name='confidence', visible=False)
    napari.run()

if __name__ == '__main__':
    # path_to_raw = Path('/groups/sgro/sgrolab/jennifer/cryolite/John/251120_1613_NC281-Fl2mSiH2B_250cryo_20XWI_volMov_imersolpt2/251120_1613_NC281-Fl2mSiH2B_250cryo_20XWI_volMov_imersolpt2.zarr')
    path_to_raw = Path('Y:\\jennifer\\cryolite\\John\\251120_1613_NC281-Fl2mSiH2B_250cryo_20XWI_volMov_imersolpt2\\251120_1613_NC281-Fl2mSiH2B_250cryo_20XWI_volMov_imersolpt2.zarr')
    # path_to_raw = Path('/Volumes/sgrolab/jennifer/cryolite/John/251120_1613_NC281-Fl2mSiH2B_250cryo_20XWI_volMov_imersolpt2/251120_1613_NC281-Fl2mSiH2B_250cryo_20XWI_volMov_imersolpt2.zarr')
    flow_exp = '2026-01-08_15-37-49'
    main(path_to_raw, flow_exp, compute=True)