import numpy as np
from skimage.filters import gaussian
from skimage.filters import threshold_otsu as sk_threshold_otsu
from skimage.measure import label
from skimage.morphology import local_maxima
from skimage.segmentation import watershed


def voronoi_otsu_labeling(image, spot_sigma: float = 2, outline_sigma: float = 1):
    """Copied from napari-segment-blobs-and-things-with-membranes
    https://github.com/haesleinhuepf/napari-segment-blobs-and-things-with-membranes/blob/2f04ac40383ff1c7df390e21975584f6dad64c11/napari_segment_blobs_and_things_with_membranes/__init__.py#L516
    Voronoi-Otsu-Labeling is a segmentation algorithm for blob-like structures such as
    nuclei and granules with high signal intensity on low-intensity background.

    Args:
        image (np.ndarrray): _description_
        spot_sigma (float, optional): Controls how close detected cells can be by
            smoothing before detecting local maxima to use as watershed seeds.
            Defaults to 2.
        outline_sigma (float, optional): Controls how precise segmented objects are
            outlined by smoothing before performing otsu thresholding to get fg/bg mask.
            Defaults to 1.

    Returns:
        np.ndarray: Labels array of same shape as input and dtype int32.
    """
    image = np.asarray(image)

    # blur and detect local maxima
    blurred_spots = gaussian(image, spot_sigma)
    spot_centroids = local_maxima(blurred_spots)

    # blur and threshold
    blurred_outline = gaussian(image, outline_sigma)
    threshold = sk_threshold_otsu(blurred_outline)
    binary_otsu = blurred_outline > threshold

    # determine local maxima within the thresholded area
    remaining_spots = spot_centroids * binary_otsu

    # start from remaining spots and flood binary image with labels
    labeled_spots = label(remaining_spots)
    labels = watershed(binary_otsu, labeled_spots, mask=binary_otsu)

    return labels
