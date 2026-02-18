import napari
import zarr
import dask.array as da


def main(raw_zarr_path, seg_zarr_path, compute=False):
    raw = da.from_zarr(raw_zarr_path)
    raw = raw[:, 0, ...]

    seg = zarr.open(seg_zarr_path, mode='r')
    affinities = da.from_zarr(seg['affinities'])[:, ...]
    fragments = da.from_zarr(seg['fragments'])[:, ...]
    segmentations = da.from_zarr(seg['segmentations'])[:, ...]
    axes = seg['affinities'].attrs.get("axes", None)
    if axes is not None:
        scale = [axis["scale"] for axis in axes if axis["scale"] is not None]
    else:
        scale = [1.0, 1.0, 1.0]

    print(f"Raw shape: {raw.shape}, dtype: {raw.dtype}")
    print(f"Affinities shape: {affinities.shape}, dtype: {affinities.dtype}")
    print(f"Fragments shape: {fragments.shape}, dtype: {fragments.dtype}")
    print(f"Segmentations shape: {segmentations.shape}, dtype: {segmentations.dtype}")

    if compute:
        raw = raw.compute()
        affinities = affinities.compute()
        fragments = fragments.compute()
        segmentations = segmentations.compute()

    affinities = 1.0 - affinities

    viewer = napari.Viewer()
    viewer.add_image(raw, name='raw', scale=scale)
    viewer.add_image(affinities, name='affinities', channel_axis=1, contrast_limits=[0, 1], scale=scale)
    viewer.add_labels(fragments, name='fragments', scale=scale)
    viewer.add_labels(segmentations, name='segmentations', opacity=0.5, scale=scale)

    napari.run()

if __name__ == '__main__':
    # raw_zarr_path = '/groups/sgro/sgrolab/jennifer/mhat/data/NC281-Fl2mSiH2B/02_cells.zarr'
    # seg_zarr_path = '/groups/sgro/sgrolab/jennifer/mhat/experiments/segmentation/NC281-Fl2mSiH2B/03_test_data/data.zarr'
    raw_zarr_path = 'Y:\\jennifer\\mhat\\data\\Fluo-C3DL-MDA231\\01_cells.zarr'
    seg_zarr_path = 'Y:\\jennifer\\mhat\\experiments\\segmentation\\Fluo-C3DL-MDA231\\01_cells\\2026-02-18_16-09-11\\data.zarr'
    # raw_zarr_path = '/Volumes/sgrolab/jennifer/mhat/data/mixin63/02_test_data.zarr'
    # seg_zarr_path = '/Volumes/sgrolab/jennifer/mhat/experiments/segmentation/mixin63/03_test_data/data.zarr'
    main(raw_zarr_path, seg_zarr_path, compute=True)
