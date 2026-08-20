from typing import Optional

import numpy as np

from constants import n, q, field


class PolyRingRqNtt:
    def __init__(
        self,
        zeta: Optional[int] = None,
        ntt_matrix: Optional[np.ndarray] = None,
        intt_matrix: Optional[np.ndarray] = None,
    ):
        self.zeta = zeta

        if self.zeta is None:
            self.zeta = find_zeta(n, q)

        if ntt_matrix is None:
            ntt_matrix = find_ntt_matrix(n, q, self.zeta)
        self.ntt_matrix = field(ntt_matrix)

        if intt_matrix is None:
            intt_matrix = find_intt_matrix(n, q, self.zeta)
        self.intt_matrix = field(intt_matrix)

    def ntt(self, A: np.ndarray) -> np.ndarray:
        shape = A.shape
        A_flat = A.reshape(n, -1)
        A_hat_flat = self.ntt_matrix @ A_flat
        A_hat = A_hat_flat.reshape(shape)
        return A_hat

    def intt(self, A_hat: np.ndarray) -> np.ndarray:
        shape = A_hat.shape
        A_hat_flat = A_hat.reshape(n, -1)
        A_flat = self.intt_matrix @ A_hat_flat
        A = A_flat.reshape(shape)
        return A


def find_zeta(n: int, q: int) -> int:
    """Util to find a valid zeta (which can then be hardcoded)."""

    k = (q - 1) // (2 * n)
    zeta = 1
    while pow(zeta, n, q) != q - 1:
        a = np.random.randint(2, q - 1)
        zeta = pow(a, k, q)
    return zeta


def find_ntt_matrix(n: int, q: int, zeta: int) -> np.ndarray:
    """Util to find a valid NTT matrix (which can then be hardcoded)."""

    ntt_matrix = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for k in range(n):
            ntt_matrix[i, k] = pow(zeta, (2 * i + 1) * k, q)
    return ntt_matrix


def find_intt_matrix(n: int, q: int, zeta: int) -> np.ndarray:
    """Util to find a valid INTT matrix (which can then be hardcoded)."""

    zeta_inv = pow(zeta, -1, q)
    n_inv = pow(n, -1, q)

    intt_matrix = np.zeros((n, n), dtype=np.int64)
    for j in range(n):
        for i in range(n):
            product = pow(zeta_inv, j, q)
            product = (product * pow(zeta_inv, 2 * i * j, q)) % q
            product = (product * n_inv) % q
            intt_matrix[j, i] = product

    return intt_matrix
