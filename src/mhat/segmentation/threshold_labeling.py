import numpy as np
from skimage.filters import gaussian
from skimage.filters import threshold_otsu
from skimage.filters import threshold_mean
from skimage.filters import threshold_li
from skimage.filters import threshold_triangle
from skimage.measure import label
from skimage.morphology import local_maxima
from skimage.segmentation import watershed


# Threshold methods that derive a scalar cutoff from the (blurred) image.
# "fixed" uses a user-supplied value; the rest are computed from the data.
THRESHOLD_METHODS = ("fixed", "otsu", "mean", "li", "triangle", "mad")


def compute_threshold(image, method: str, threshold: float = None, mad_k: float = 9.0):
    """Compute a scalar foreground threshold for ``image`` via the named method.

    Args:
        image (np.ndarray): Image to threshold (typically already blurred).
        method (str): One of ``THRESHOLD_METHODS``.
            - "fixed": return the user-supplied ``threshold``.
            - "otsu"/"mean"/"li"/"triangle": skimage automatic thresholds.
            - "mad": robust background estimate, ``median + mad_k * (1.4826 * MAD)``,
              where MAD is the median absolute deviation. Immune to the sparse-
              foreground class-imbalance failure that pulls otsu/li onto the
              background peak.
        threshold (float, optional): Cutoff value; required for method="fixed".
        mad_k (float, optional): Multiplier on the robust background sigma for
            method="mad". Defaults to 9.0.

    Returns:
        float: The threshold value.
    """
    if method == "fixed":
        if threshold is None:
            raise ValueError("A `threshold` value must be provided when method='fixed'.")
        return threshold
    if method == "otsu":
        return threshold_otsu(image)
    if method == "mean":
        return threshold_mean(image)
    if method == "li":
        return threshold_li(image)
    if method == "triangle":
        return threshold_triangle(image)
    if method == "mad":
        median = np.median(image)
        mad_std = np.median(np.abs(image - median)) * 1.4826
        return median + mad_k * mad_std
    raise ValueError(
        f"Unknown threshold method {method!r}. Choose from {THRESHOLD_METHODS}."
    )


def threshold_labeling(
    image,
    method: str = "fixed",
    spot_sigma: float = 2,
    outline_sigma: float = 1,
    threshold: float = None,
    mad_k: float = 9.0,
):
    """Voronoi-Otsu-style labeling with a configurable thresholding method.

    Blurs the image to find local maxima (watershed seeds), thresholds a second
    blurred copy to get a foreground mask, and floods the seeds within that mask
    via watershed. The only thing that varies between methods is how the
    foreground/background cutoff is chosen (see ``compute_threshold``).

    Args:
        image (np.ndarray): Input image.
        method (str, optional): Thresholding method, one of ``THRESHOLD_METHODS``.
            Defaults to "fixed".
        spot_sigma (float, optional): Gaussian sigma for seed detection.
            Defaults to 2.
        outline_sigma (float, optional): Gaussian sigma for the foreground mask.
            Defaults to 1.
        threshold (float, optional): Cutoff value; required for method="fixed".
        mad_k (float, optional): Multiplier on the robust background sigma for
            method="mad". Defaults to 9.0.

    Returns:
        np.ndarray: Integer label image of the same shape as ``image``.
    """
    image = np.asarray(image)

    # blur and detect local maxima to use as watershed seeds
    blurred_spots = gaussian(image, spot_sigma)
    spot_centroids = local_maxima(blurred_spots)

    # blur and threshold to get the foreground mask
    blurred_outline = gaussian(image, outline_sigma)
    thresh = compute_threshold(blurred_outline, method, threshold=threshold, mad_k=mad_k)
    binary = blurred_outline > thresh

    # keep only local maxima that fall within the thresholded area
    remaining_spots = spot_centroids * binary

    # start from remaining spots and flood binary image with labels
    labeled_spots = label(remaining_spots)
    labels = watershed(binary, labeled_spots, mask=binary)

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
