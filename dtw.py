import numpy as np
import numba as nb
from typing import Tuple
from math import sqrt


@nb.njit((nb.float32[:, :])(nb.float32[:, ::2], nb.float32[:, ::2]))
def distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    D = np.empty((x.shape[0], y.shape[0]), dtype=np.float32)
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            dx = x[i, 0] - y[j, 0]
            dy = x[i, 1] - y[j, 1]
            D[i, j] = dx * dx + dy * dy
    return D


@nb.njit(nb.types.Tuple((nb.float32, nb.int64[:, ::2]))(nb.float32[:, ::2], nb.float32[:, ::2]))
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
    pairs = np.empty((D.shape[0] + D.shape[1] - 1, 2), dtype=np.int64)  # size <= m + n - 1
    pairs_size = 0
    i = D.shape[0] - 1
    j = D.shape[1] - 1
    while i >= 0 and j >= 0:
        pairs[pairs_size, 0] = i
        pairs[pairs_size, 1] = j
        pairs_size += 1
        if previous[i, j] == 1:
            i -= 1
        elif previous[i, j] == 2:
            j -= 1
        else:
            i -= 1
            j -= 1
    return D[-1, -1], pairs[:pairs_size]


# Matrica s udaljenostima crteža iz prvog niza od crteža iz drugog niza
@nb.njit(parallel=True)
def custom_distance_matrix(drawings1: np.ndarray, drawings2: np.ndarray, distance_function) -> np.ndarray:
    m = drawings1.shape[0]
    n = drawings2.shape[0]
    D = np.empty((m, n), dtype=np.float32)
    for i in nb.prange(m):  # Paralelno računanje
        for j in range(n):
            D[i, j] = distance_function(drawings1[i], drawings2[j])
    return D


# Matrica s udaljenostima crteža iz prvog niza od crteža iz drugog niza
# Ova funkcija točno računa samo ako prvi niz počinje s drugim nizom, jer samo jednom računa udaljenosti između crteža iz drugog niza.
# To je je moguće za svaku simetričnu funkciju udaljenosti
@nb.njit(parallel=True)
def custom_distance_matrix_symmetric(drawings1: np.ndarray, drawings2: np.ndarray, distance_function) -> np.ndarray:
    m = drawings1.shape[0]
    n = drawings2.shape[0]
    D = np.empty((m, n), dtype=np.float32)
    # Udaljenosti između crtaža iz drugog niza:
    for i in nb.prange(n):
        D[i, i] = 0
        for j in range(i + 1, n):
            d = distance_function(drawings1[i], drawings1[j])
            D[i, j] = d
            D[j, i] = d
    # Udaljenosti izmežu ostalih crteža:
    for i in nb.prange(n, m):
        for j in range(n):
            D[i, j] = distance_function(drawings1[i], drawings1[j])
    return D


@nb.njit(nb.float32(nb.float32[:, ::2], nb.float32[:, ::2]))
def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    return dtw(x, y)[0]


@nb.njit(nb.float32[:, :](nb.float32[:, :, ::2], nb.float32[:, :, ::2]))
def dtw_distance_matrix(drawings1: np.ndarray, drawings2: np.ndarray) -> np.ndarray:
    return custom_distance_matrix(drawings1, drawings2, dtw_distance)


@nb.njit(nb.float32[:, :](nb.float32[:, :, ::2], nb.float32[:, :, ::2]))
def dtw_distance_matrix_symmetric(drawings1: np.ndarray, drawings2: np.ndarray) -> np.ndarray:
    return custom_distance_matrix_symmetric(drawings1, drawings2, dtw_distance)


# Najbolja kombinacija rotacije i translacije prvog niza, koja se najbolje poravnava s drugim nizom
@nb.njit(nb.types.Tuple((nb.float32[::2, ::2], nb.float32[::2]))(nb.float32[:, ::2], nb.float32[:, ::2]))
def rs_alignment(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    # Prosjeci koordinata
    xx = np.empty(2, dtype=np.float32)
    yy = np.empty(2, dtype=np.float32)
    xx[0] = x[:, 0].mean()
    xx[1] = x[:, 1].mean()
    yy[0] = y[:, 0].mean()
    yy[1] = y[:, 1].mean()
    x = x - xx
    y = y - yy
    # Izračun optimalnog kuta rotacije
    c = 0
    d = 0
    for i in range(n):
        c += x[i, 0] * y[i, 0]
    for i in range(n):
        c += x[i, 1] * y[i, 1]
    for i in range(n):
        d += x[i, 0] * y[i, 1]
    for i in range(n):
        d -= x[i, 1] * y[i, 0]
    if c < 0:
        d = -d
        c = -c
    hyp = sqrt(c * c + d * d)
    sin_theta = d / hyp
    cos_theta = c / hyp
    # Transponirana matrica rotacije
    rotation = np.empty((2, 2), dtype=np.float32)
    rotation[0, 0] = cos_theta
    rotation[0, 1] = sin_theta
    rotation[1, 0] = -sin_theta
    rotation[1, 1] = cos_theta
    return rotation, yy - xx @ rotation


# Matrično množenje s 2x2 matricom (numba nekad baca upozorenja za operator @)
@nb.njit(nb.float32[:, ::2](nb.float32[:, ::2], nb.float32[::2, ::2], nb.float32[:, ::2]))
def matmul22(x: np.ndarray, y: np.ndarray, result: np.ndarray):
    result[:, 0] = x[:, 0] * y[0, 0] + x[:, 1] * y[1, 0]
    result[:, 1] = x[:, 0] * y[0, 1] + x[:, 1] * y[1, 1]
    return result


# Množenje retka s 2x2 matricom
@nb.njit(nb.float32[::2](nb.float32[::2], nb.float32[::2, ::2], nb.float32[::2]))
def vecmul22(x: np.ndarray, y: np.ndarray, result: np.ndarray):
    result[0] = x[0] * y[0, 0] + x[1] * y[1, 0]
    result[1] = x[0] * y[0, 1] + x[1] * y[1, 1]
    return result


# Rotation-Scale Invariant DTW
@nb.njit(nb.types.Tuple(
    (nb.float32, nb.int64[:, ::2], nb.float32[::2, ::2], nb.float32[::2])
)(nb.float32[:, ::2], nb.float32[:, ::2], nb.int64))
def rsi_dtw(x: np.ndarray, y: np.ndarray, n_iter: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    # Konačna transformacija
    rotation = np.eye(2, dtype=np.float32)
    shift = np.zeros(2, dtype=np.float32)
    # Transformirani x
    x_ = x.copy()
    # Za spremanje umnožaka s matricom
    rotation_ = np.empty_like(rotation)
    shift_ = np.empty_like(shift)
    for i in range(n_iter):
        _, matching = dtw(x_, y)
        new_rotation, new_shift = rs_alignment(x_[matching[:, 0]], y[matching[:, 1]])
        # Update konačne transformacije
        rotation[:] = matmul22(rotation, new_rotation, rotation_)
        shift = vecmul22(shift, new_rotation, shift_) + new_shift
        matmul22(x, rotation, x_)
        x_ += shift
    distance, matching = dtw(x_, y)
    return distance, matching, rotation, shift


def make_rsi_dtw_distance(n_iter: int):
    # Funckija udaljenosti za fiksni n_iter
    @nb.njit(nb.float32(nb.float32[:, ::2], nb.float32[:, ::2]))
    def rsi_dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
        return rsi_dtw(x, y, n_iter)[0]

    return rsi_dtw_distance


def rsi_dtw_distance_matrix(drawings1: np.ndarray, drawings2: np.ndarray, n_iter: int) -> np.ndarray:
    return custom_distance_matrix(drawings1, drawings2, make_rsi_dtw_distance(n_iter))


def rsi_dtw_distance_matrix_symmetric(drawings1: np.ndarray, drawings2: np.ndarray, n_iter: int) -> np.ndarray:
    return custom_distance_matrix_symmetric(drawings1, drawings2, make_rsi_dtw_distance(n_iter))
