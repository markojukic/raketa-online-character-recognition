import numpy as np
import numba as nb
from typing import Tuple


@nb.njit((nb.float32[:, :])(nb.float32[:, ::2], nb.float32[:, ::2]))
def distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    D = np.empty((x.shape[0], y.shape[0]), dtype=np.float32)
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            dx = x[i, 0] - y[j, 0]
            dy = x[i, 1] - y[j, 1]
            D[i, j] = dx * dx + dy * dy
    return D


@nb.njit(nb.types.Tuple((nb.float32, nb.int32[:, ::2]))(nb.float32[:, ::2], nb.float32[:, ::2]))
def dtw(x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    D = distance_matrix(x, y)
    previous = np.empty_like(D, dtype=np.int8)
    previous[0, 0] = 3
    for i in range(1, D.shape[0]):
        D[i, 0] += D[i - 1, 0]
        previous[i, 0] = 1
    for j in range(1, D.shape[1]):
        D[0, j] += D[0, j - 1]
        previous[0, j] = 2
    for i in range(1, D.shape[0]):
        for j in range(1, D.shape[1]):
            up = D[i - 1, j]
            left = D[i, j - 1]
            up_left = D[i - 1, j - 1]
            if up <= left and up <= up_left:
                D[i][j] += up
                previous[i][j] = 1
            elif left <= up and left <= up_left:
                D[i][j] += left
                previous[i][j] = 2
            else:
                D[i][j] += up_left
                previous[i][j] = 3
    # Backtracking
    pairs = np.empty((D.shape[0] + D.shape[1] - 1, 2), dtype=np.int32)  # size <= m + n - 1
    pairs_size = 0
    i = D.shape[0] - 1
    j = D.shape[1] - 1
    while i >= 0 and j >= 0:
        pairs[pairs_size] = (i, j)
        pairs_size += 1
        if previous[i, j] & 1:  # 1 ili 3
            i -= 1
        if previous[i, j] & 2:  # 2 ili 3
            j -= 1
    return D[-1, -1], pairs[:pairs_size]


@nb.njit(nb.float32[:, :](nb.float32[:, :, ::2]), parallel=True)
def dtw_distance_matrix(drawings: np.ndarray) -> np.ndarray:
    n = drawings.shape[0]
    D = np.empty((n, n), dtype=np.float32)
    for i in nb.prange(n):
        D[i, i] = 0
        for j in range(i + 1, n):
            d, _ = dtw(drawings[i], drawings[j])
            D[i, j] = d
            D[j, i] = d
    return D
