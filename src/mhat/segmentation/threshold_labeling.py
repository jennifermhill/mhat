import numpy as np
from skimage.filters import gaussian
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import local_maxima
from skimage.segmentation import watershed
from skimage.filters import threshold_mean
from skimage.filters import threshold_li


def voronoi_otsu_labeling(image, spot_sigma: float = 2, outline_sigma: float = 1):
    """Copied from napari-segment-blobs-and-things-with-membranes
    https://github.com/haesleinhuepf/napari-segment-blobs-and-things-with-membranes/blob/2f04ac40383ff1c7df390e21975584f6dad64c11/napari_segment_blobs_and_things_with_membranes/__init__.py#L516
    Voronoi-Otsu-Labeling is a segmentation algorithm for blob-like structures such as
    nuclei and granules with high signal intensity on low-intensity background.

    Args:
        image (np.ndarrray): Input image.
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
    threshold = threshold_otsu(blurred_outline)
    
    binary_otsu = blurred_outline > threshold

    # determine local maxima within the thresholded area
    remaining_spots = spot_centroids * binary_otsu

    # start from remaining spots and flood binary image with labels
    labeled_spots = label(remaining_spots)
    labels = watershed(binary_otsu, labeled_spots, mask=binary_otsu)

    return labels

def voronoi_mean_labeling(image, spot_sigma: float = 2, outline_sigma: float = 1):
    """Simple segmentation algorithm that thresholds the image at its mean intensity.

    Args:
        image (np.ndarray): Input image.
        spot_sigma (float, optional): Unused parameter for compatibility.
            Defaults to 2.
        outline_sigma (float, optional): Unused parameter for compatibility.
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
    threshold = threshold_mean(blurred_outline)

    binary_mean = blurred_outline > threshold

    # determine local maxima within the thresholded area
    remaining_spots = spot_centroids * binary_mean

    # start from remaining spots and flood binary image with labels
    labeled_spots = label(remaining_spots)
    labels = watershed(binary_mean, labeled_spots, mask=binary_mean)

    return labels

def voronoi_li_labeling(image, spot_sigma: float = 2, outline_sigma: float = 1):
    """Simple segmentation algorithm that thresholds the image using the Li thresholding method.

    Args:
        image (np.ndarray): Input image.
        spot_sigma (float, optional): Unused parameter for compatibility.
            Defaults to 2.
        outline_sigma (float, optional): Unused parameter for compatibility.
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
    threshold = threshold_li(blurred_outline)

    binary_li = blurred_outline > threshold

    # determine local maxima within the thresholded area
    remaining_spots = spot_centroids * binary_li

    # start from remaining spots and flood binary image with labels
    labeled_spots = label(remaining_spots)
    labels = watershed(binary_li, labeled_spots, mask=binary_li)

    return labels


def segment_cells_from_nuclei_frame(cell_img, nuclei_labels_img):
    """Segment cells in a single frame using nuclei centroids as watershed seeds.

    Args:
        cell_img (np.ndarray): Raw cell-channel image for one timepoint, (z, y, x).
        nuclei_labels_img (np.ndarray): Nuclei label image for the same timepoint,
            (z, y, x). Output cell labels match these nuclei labels.

    Returns:
        np.ndarray: Cell label image, same shape as the inputs.
    """
    from skimage.measure import regionprops

    nuclei_props = regionprops(nuclei_labels_img)
    if len(nuclei_props) == 0:
        return np.zeros_like(nuclei_labels_img)

    centroids = np.array([prop.centroid for prop in nuclei_props]).astype(int)
    labels = np.array([prop.label for prop in nuclei_props])

    # Blur and threshold the raw cell image to get a binary mask
    blurred = gaussian(cell_img, sigma=1)
    threshold = threshold_otsu(blurred)
    binary_mask = blurred > threshold

    # Build marker array from centroids, dropping any outside the mask
    markers = np.zeros_like(nuclei_labels_img)
    for centroid, label in zip(centroids, labels):
        coord = tuple(centroid)
        if binary_mask[coord]:
            markers[coord] = label

    # Expand evenly from seeds (Voronoi partition within mask)
    return watershed(np.zeros_like(cell_img), markers, mask=binary_mask)


def segment_cells_from_nuclei(raw_cell_img, nuclei_seg):
    """Segment cells across all timepoints from nuclei centroid seeds.

    Builds the full segmentation in memory. For large datasets, drive
    ``segment_cells_from_nuclei_frame`` in a loop that writes each frame to disk
    instead of materializing the whole volume.
    """
    cell_seg = np.zeros_like(nuclei_seg)
    for t in range(nuclei_seg.shape[0]):
        cell_seg[t] = segment_cells_from_nuclei_frame(raw_cell_img[t], nuclei_seg[t])

    return cell_seg
