from pathlib import Path
import napari
import zarr
import dask.array as da
from mhat.opticalflow.visualization import generate_flow_frames

def main(path_to_raw: Path, flow_exp: str, compute: bool = False):
    raw_img = da.from_zarr(path_to_raw / '0' / '0')
    print(f"raw img shape: {raw_img.shape}")

    raw_img = raw_img[:, 3, 33, ...]
    # Downsample to match optical flow resolution
    raw_img = raw_img[:, ::2, ::2]
    print(f"raw img shape after slicing and downsampling: {raw_img.shape}")

    flow_exp_path = path_to_raw.parent / 'opticalflow_2d' / flow_exp / 'flow.zarr'
    if not flow_exp_path.exists():
        raise FileNotFoundError(f"Flow experiment path does not exist: {flow_exp_path}")
    if not (flow_exp_path / 'flow_frames').exists():
        print(f"Flow frames path does not exist. Creating at: {flow_exp_path / 'flow_frames'}")
        flow_zarr = zarr.open(flow_exp_path, mode='a')
        generate_flow_frames(flow_zarr, scale_factor=1, color_wheel=True)
    flow_frames = da.from_zarr(flow_exp_path / 'flow_frames')
    print(f"flow frames shape: {flow_frames.shape}")

    if compute:
        print("Computing frames from dask arrays...")
        raw_img = raw_img.compute()
        flow_frames = flow_frames.compute()
        print("Computation complete.")

    viewer = napari.Viewer()
    viewer.add_image(raw_img, name='raw')
    viewer.add_image(flow_frames, name='flow frames', blending='additive')
    napari.run()

if __name__ == '__main__':
    # path_to_raw = Path('/groups/sgro/sgrolab/jennifer/cryolite/John/251120_1613_NC281-Fl2mSiH2B_250cryo_20XWI_volMov_imersolpt2/251120_1613_NC281-Fl2mSiH2B_250cryo_20XWI_volMov_imersolpt2.zarr')
    path_to_raw = Path('Y:\\jennifer\\cryolite\\John\\251120_1613_NC281-Fl2mSiH2B_250cryo_20XWI_volMov_imersolpt2\\251120_1613_NC281-Fl2mSiH2B_250cryo_20XWI_volMov_imersolpt2.zarr')
    # path_to_raw = Path('/Volumes/sgrolab/jennifer/cryolite/John/251120_1613_NC281-Fl2mSiH2B_250cryo_20XWI_volMov_imersolpt2/251120_1613_NC281-Fl2mSiH2B_250cryo_20XWI_volMov_imersolpt2.zarr')
    flow_exp = '2026-01-14_14-34-59'
    main(path_to_raw, flow_exp, compute=True)