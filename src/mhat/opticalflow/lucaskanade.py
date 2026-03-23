import sys
import math
import torch
from datetime import datetime
import dask.array as da
import numpy as np
from tqdm import tqdm
from scipy.ndimage import zoom

from mhat.opticalflow.utils import enhance_contrast_AHE, correlate1d_gpu, get_device

def calc_flow3D(images, xyzSig=3, tSig=1, wSig=4, device=None):
    """
    Calculate three-dimensional optical flow fields from input z-stacks (GPU).

    Same API and behavior as calc_flow.calc_flow3D, but runs on GPU.

    ARGS:
    images: 4D numpy array with dimensions N_T, N_Z, N_Y, N_X
    xyzSig: sigma for spatial smoothing. Default 3.
    tSig:   sigma for temporal smoothing. Default 1.
    wSig:   sigma for Lucas-Kanade neighborhood. Default 4.
    device: torch device string or None for auto-detect.

    RETURNS:
    vx, vy, vz, rel as numpy arrays (same as CPU version).
    """
    device = get_device(device)

    ### Check inputs ###########################################################
    if len(images.shape) != 4:
        sys.exit('ERROR: Input image must be a 4D matrix with dimensions N_T, N_Z, N_Y, N_X')
    Nt = images.shape[0]
    if Nt < 6 * tSig + 1:
        sys.exit('ERROR: Input images will lead to edge effects. N_T must be >= 6*tSig+1')
    if not (Nt % 2):
        sys.exit('ERROR: Input images must have an odd number of timepoints. Only the central time point is analyzed')
    NtSlice = math.ceil(Nt / 2) - 1

    # Transfer to GPU as float32
    images = torch.as_tensor(images, dtype=torch.float32, device=device)

    ### Set up filters #########################################################
    x = np.arange(-math.ceil(3 * xyzSig), math.ceil(3 * xyzSig) + 1)
    xyzSig2 = xyzSig / 4
    y = np.arange(-math.ceil(3 * xyzSig2), math.ceil(3 * xyzSig2) + 1)
    fderiv = np.exp(-x * x / 2 / xyzSig / xyzSig) / math.sqrt(2 * math.pi) / xyzSig
    fsmooth = np.exp(-y * y / 2 / xyzSig2 / xyzSig2) / math.sqrt(2 * math.pi) / xyzSig2
    gderiv = x / xyzSig / xyzSig
    gsmooth = 1

    # y-gradient kernels
    yFil1 = torch.tensor(fderiv * gderiv, dtype=torch.float32, device=device)
    xFil1 = torch.tensor(fsmooth * gsmooth, dtype=torch.float32, device=device)
    zFil1 = torch.tensor(fsmooth * gsmooth, dtype=torch.float32, device=device)
    # x-gradient kernels
    yFil2 = torch.tensor(fsmooth * gsmooth, dtype=torch.float32, device=device)
    xFil2 = torch.tensor(fderiv * gderiv, dtype=torch.float32, device=device)
    zFil2 = torch.tensor(fsmooth * gsmooth, dtype=torch.float32, device=device)
    # z-gradient kernels
    yFil3 = torch.tensor(fsmooth * gsmooth, dtype=torch.float32, device=device)
    xFil3 = torch.tensor(fsmooth * gsmooth, dtype=torch.float32, device=device)
    zFil3 = torch.tensor(fderiv * gderiv, dtype=torch.float32, device=device)

    # t-gradient kernels
    t = np.arange(-math.ceil(3 * tSig), math.ceil(3 * tSig) + 1)
    fx = np.exp(-x * x / 2 / xyzSig / xyzSig) / math.sqrt(2 * math.pi) / xyzSig
    ft = np.exp(-t * t / 2 / tSig / tSig) / math.sqrt(2 * math.pi) / tSig
    gx = 1
    gt = t / tSig / tSig
    yFil4 = torch.tensor(fx * gx, dtype=torch.float32, device=device)
    xFil4 = yFil4.clone()
    zFil4 = yFil4.clone()
    tFil4 = torch.tensor(ft * gt, dtype=torch.float32, device=device)

    # LK neighborhood kernels
    wRange = np.arange(-math.ceil(3 * wSig), math.ceil(3 * wSig) + 1)
    gw = np.exp(-wRange * wRange / 2 / wSig / wSig) / math.sqrt(2 * math.pi) / wSig
    yFil5 = torch.tensor(gw, dtype=torch.float32, device=device)
    xFil5 = yFil5.clone()
    zFil5 = yFil5.clone()

    ### Spatial and Temporal Gradients ##########################################
    # Temporal gradient on full 4D array, then slice center frame
    # axis 0 = time in 4D (N_T, N_Z, N_Y, N_X)
    dtI = correlate1d_gpu(images, tFil4, axis=0)
    dtI = dtI[NtSlice]
    images = images[NtSlice]
    # Now dtI and images are 3D: (N_Z, N_Y, N_X)
    # axes: 0=z, 1=y, 2=x
    dtI = correlate1d_gpu(correlate1d_gpu(correlate1d_gpu(dtI, yFil4, axis=1), xFil4, axis=2), zFil4, axis=0)
    del xFil4, yFil4, zFil4, tFil4
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    dyI = correlate1d_gpu(correlate1d_gpu(correlate1d_gpu(images, yFil1, axis=1), xFil1, axis=2), zFil1, axis=0)
    del xFil1, yFil1, zFil1

    dxI = correlate1d_gpu(correlate1d_gpu(correlate1d_gpu(images, yFil2, axis=1), xFil2, axis=2), zFil2, axis=0)
    del xFil2, yFil2, zFil2

    dzI = correlate1d_gpu(correlate1d_gpu(correlate1d_gpu(images, yFil3, axis=1), xFil3, axis=2), zFil3, axis=0)
    del xFil3, yFil3, zFil3
    del images
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    ### Structure Tensor Inputs ################################################
    # Time components
    wdtx = correlate1d_gpu(correlate1d_gpu(correlate1d_gpu(dxI * dtI, yFil5, axis=1), xFil5, axis=2), zFil5, axis=0)
    wdty = correlate1d_gpu(correlate1d_gpu(correlate1d_gpu(dyI * dtI, yFil5, axis=1), xFil5, axis=2), zFil5, axis=0)
    wdtz = correlate1d_gpu(correlate1d_gpu(correlate1d_gpu(dzI * dtI, yFil5, axis=1), xFil5, axis=2), zFil5, axis=0)
    del dtI
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Spatial components
    wdxy = correlate1d_gpu(correlate1d_gpu(correlate1d_gpu(dxI * dyI, yFil5, axis=1), xFil5, axis=2), zFil5, axis=0)
    wdxz = correlate1d_gpu(correlate1d_gpu(correlate1d_gpu(dxI * dzI, yFil5, axis=1), xFil5, axis=2), zFil5, axis=0)
    wdx2 = correlate1d_gpu(correlate1d_gpu(correlate1d_gpu(dxI * dxI, yFil5, axis=1), xFil5, axis=2), zFil5, axis=0)
    del dxI
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    wdyz = correlate1d_gpu(correlate1d_gpu(correlate1d_gpu(dyI * dzI, yFil5, axis=1), xFil5, axis=2), zFil5, axis=0)
    wdy2 = correlate1d_gpu(correlate1d_gpu(correlate1d_gpu(dyI * dyI, yFil5, axis=1), xFil5, axis=2), zFil5, axis=0)
    del dyI
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    wdz2 = correlate1d_gpu(correlate1d_gpu(correlate1d_gpu(dzI * dzI, yFil5, axis=1), xFil5, axis=2), zFil5, axis=0)
    del dzI
    del xFil5, yFil5, zFil5
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    ### Optical Flow Calculations ##############################################
    eps = torch.finfo(torch.float32).eps
    determinant = (wdx2 * wdy2 * wdz2) + (2 * wdxy * wdxz * wdyz) - (wdy2 * wdxz ** 2) - (wdz2 * wdxy ** 2) - (wdx2 * wdyz ** 2)
    inv_det = (determinant + eps).reciprocal()
    del determinant

    vx = -inv_det * ((wdy2 * wdz2 - wdyz * wdyz) * wdtx + (wdxz * wdyz - wdxy * wdz2) * wdty + (wdxy * wdyz - wdxz * wdy2) * wdtz)
    vy = -inv_det * ((wdyz * wdxz - wdxy * wdz2) * wdtx + (wdx2 * wdz2 - wdxz * wdxz) * wdty + (wdxz * wdxy - wdx2 * wdyz) * wdtz)
    vz = -inv_det * ((wdxy * wdyz - wdy2 * wdxz) * wdtx + (wdxy * wdxz - wdx2 * wdyz) * wdty + (wdx2 * wdy2 - wdxy * wdxy) * wdtz)

    del wdtx, wdty, wdtz, inv_det
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    ### Eigenvalues for Reliability (analytical) ###############################
    # 3x3 symmetric matrix with upper triangle: a=wdx2, b=wdxy, c=wdxz, d=wdy2, e=wdyz, f=wdz2
    a, b, c, d, e, f = wdx2, wdxy, wdxz, wdy2, wdyz, wdz2
    del wdx2, wdxy, wdxz, wdy2, wdyz, wdz2

    p1 = b * b + c * c + e * e  # sum of squares of off-diagonal elements

    # trace / 3
    q = (a + d + f) / 3.0

    # p2 = sum of squares of (A - qI) diagonal + 2*p1
    p2 = (a - q) ** 2 + (d - q) ** 2 + (f - q) ** 2 + 2.0 * p1
    p = torch.sqrt(p2 / 6.0)

    # For numerical stability, handle the case where p is ~0 (diagonal matrix)
    # We'll compute both branches and select
    p_safe = torch.clamp(p, min=1e-30)

    # Compute det((A - qI) / p) for the general case
    # (A - qI) / p diagonal: (a-q)/p, (d-q)/p, (f-q)/p
    # off-diagonal: b/p, c/p, e/p
    a_s = (a - q) / p_safe
    d_s = (d - q) / p_safe
    f_s = (f - q) / p_safe
    b_s = b / p_safe
    c_s = c / p_safe
    e_s = e / p_safe
    del b, c, e

    # det of 3x3 symmetric: a*d*f + 2*b*c*e - a*e^2 - d*c^2 - f*b^2
    det_B = a_s * d_s * f_s + 2 * b_s * c_s * e_s - a_s * e_s ** 2 - d_s * c_s ** 2 - f_s * b_s ** 2
    del a_s, d_s, f_s, b_s, c_s, e_s

    r = det_B / 2.0
    del det_B
    r = torch.clamp(r, -1.0, 1.0)

    phi = torch.acos(r) / 3.0
    del r

    TWO_PI_OVER_3 = 2.0 * math.pi / 3.0
    eig1 = q + 2.0 * p * torch.cos(phi)
    eig2 = q + 2.0 * p * torch.cos(phi + TWO_PI_OVER_3)
    eig3 = q + 2.0 * p * torch.cos(phi + 2.0 * TWO_PI_OVER_3)
    del phi

    # General case: minimum eigenvalue
    rel_general = torch.minimum(torch.minimum(eig1, eig2), eig3)
    del eig1, eig2, eig3

    # Diagonal case (p1 ≈ 0): eigenvalues are the diagonal elements
    rel_diagonal = torch.minimum(torch.minimum(a, d), f)
    del a, d, f

    # Select based on whether off-diagonal elements are negligible
    is_diagonal = p1 < 1e-20
    del p1, p, p_safe, p2, q
    rel = torch.where(is_diagonal, rel_diagonal, rel_general)
    del is_diagonal, rel_diagonal, rel_general

    ### Return as numpy ########################################################
    vx_np = vx.cpu().numpy()
    vy_np = vy.cpu().numpy()
    vz_np = vz.cpu().numpy()
    rel_np = rel.cpu().numpy()
    del vx, vy, vz, rel
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return vx_np, vy_np, vz_np, rel_np

def compute_lucaskanade_flow_3d(config, zarr_img, output_zarr):
    T, Z, Y, X = zarr_img.shape
    print(f"Zarr image shape: {T}, {Z}, {Y}, {X}")

    t_sig = config['t_sig']
    nb_t_chunk = 6 * t_sig + 1
    if not (nb_t_chunk % 2):
        nb_t_chunk += 1
    nb_t_slice = math.ceil(nb_t_chunk / 2) - 1

    print("Normalizing all frames...")
    max_val = zarr_img.max()
    min_val = zarr_img.min()
    frames_norm = ((zarr_img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    for hh in range(0, nb_t_slice):
        print(str(datetime.now()) + ' - No data will be saved for frame ' + str(hh) + ' to avoid edge effects')
        output_zarr['flow_raw'][hh] = np.zeros((Z, Y, X, 3), dtype=np.float32)
        output_zarr['confidence'][hh] = np.zeros((Z, Y, X), dtype=np.float32)

    for hh in range(0, T - nb_t_chunk + 1):
        loop_start = datetime.now()
        print(str(loop_start) + ' - Processing frame ' + str(hh + nb_t_slice) + '...')

        images = frames_norm[hh:hh + nb_t_chunk].compute()
        print(f"Images shape: {images.shape}")

        if config['hyperparams']['enhance_contrast']:
            for i in range(images.shape[0]):
                # TODO: ask claude to optimize this
                images[i] = enhance_contrast_AHE(images[i])

        vx, vy, vz, confidence = calc_flow3D(
            images,
            xyzSig=config['xyz_sig'], # TODO: separate into isotropic sigmas
            tSig=config['t_sig'],
            wSig=config['w_sig'],
            device=None
        )

        output_zarr['flow_raw'][hh + nb_t_slice] = np.stack((vx, vy, vz), axis=-1).astype(np.float32)
        output_zarr['confidence'][hh + nb_t_slice] = confidence.astype(np.float32)

        del images, vx, vy, vz, confidence

        frames_time = datetime.now()
        print(str(datetime.now()) + ' - Frame ' + str(hh + nb_t_slice) + ' saved.  Duration: ' + str(frames_time - loop_start))

    for hh in range(T - nb_t_slice, T):
        print(str(datetime.now()) + ' - No data will be saved for frame ' + str(hh) + ' to avoid edge effects')
        output_zarr['flow_raw'][hh] = np.zeros((Z, Y, X, 3), dtype=np.float32)
        output_zarr['confidence'][hh] = np.zeros((Z, Y, X), dtype=np.float32)

    return output_zarr['flow_raw']