import numpy as np

def compute_affinities(seg: np.ndarray, nhood: list):
    nhood = np.array(nhood)

    shape = seg.shape # Z, Y, X
    n_edges = nhood.shape[0] # 3
    dims = nhood.shape[1] # 3
    affinity = np.zeros((n_edges,) + shape, dtype=np.int32) # 3, Z, Y, X

    for e in range(n_edges):
        affinity[
            e,
            max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
            max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
            max(0, -nhood[e, 2]) : min(shape[2], shape[2] - nhood[e, 2]),
        ] = (
            (
                seg[
                    max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
                    max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
                    max(0, -nhood[e, 2]) : min(shape[2], shape[2] - nhood[e, 2]),
                ]
                == seg[
                    max(0, nhood[e, 0]) : min(shape[0], shape[0] + nhood[e, 0]),
                    max(0, nhood[e, 1]) : min(shape[1], shape[1] + nhood[e, 1]),
                    max(0, nhood[e, 2]) : min(shape[2], shape[2] + nhood[e, 2]),
                ]
            )
            * (
                seg[
                    max(0, -nhood[e, 0]) : min(shape[0], shape[0] - nhood[e, 0]),
                    max(0, -nhood[e, 1]) : min(shape[1], shape[1] - nhood[e, 1]),
                    max(0, -nhood[e, 2]) : min(shape[2], shape[2] - nhood[e, 2]),
                ]
                > 0
            )
            * (
                seg[
                    max(0, nhood[e, 0]) : min(shape[0], shape[0] + nhood[e, 0]),
                    max(0, nhood[e, 1]) : min(shape[1], shape[1] + nhood[e, 1]),
                    max(0, nhood[e, 2]) : min(shape[2], shape[2] + nhood[e, 2]),
                ]
                > 0
            )
        )

    return affinity