from .base import BaseCode
import numpy as np
from typing import Tuple
import scipy


class LinearCode(BaseCode):
    def __init__(self,
                 block_size,
                 code_size,
                 G=None,
                 H=None):
        super(LinearCode, self).__init__(block_size=block_size,
                                         code_size=code_size)

        if G is None and H is None:
            G = np.empty((block_size, code_size), dtype=np.int32)
            H = np.empty((code_size - block_size, code_size), dtype=np.int32)

        if H is None:
            P = G[:block_size, block_size:]  # block_size x (code_size - block_size)
            H = np.concatenate([P.T, np.eye(code_size - block_size, dtype=np.int32)], axis=1)  # (code_size - block_size) x code_size

        if G is None:
            P = H[:code_size - block_size, :block_size].T  # block_size x (code_size - block_size)
            G = np.concatenate([np.eye(block_size, dtype=np.int32), P], axis=1)

        self.G = G
        self.H = H
        print("G:")
        print(self.G)
        print("H:")
        print(self.H)

    def encode(self, array: np.ndarray) -> np.ndarray:
        return self.matmul(array, self.G)

    def decode(self, array: np.ndarray) -> np.ndarray:
        # TODO: table of error patterns
        raise NotImplementedError()

    def check_has_error(self, array: np.ndarray):
        e = self.matmul(self.H, array)
        print(e)
        return not np.all(e == np.zeros(self.code_size - self.block_size, dtype=np.int32))

    @staticmethod
    def bring_matrix_to_identity_residual_form(G: np.ndarray) -> Tuple[np.ndarray, bool]:
        # TODO: bring code to Cython
        for i in range(G.shape[0]):
            nz, = np.nonzero(G[i:, i])
            if nz.shape[0] == 0:
                y, x = np.nonzero(G[i:, i + 1:])
                if y.shape[0] == 0:
                    return G, False
                G[:, [i, i + 1 + x[0]]] = G[:, [i + 1 + x[0], i]]
            nz, = np.nonzero(G[i:, i])
            idx = i + nz[0]
            G[[i, idx]] = G[[idx, i]]
            nz = np.concatenate([np.nonzero(G[:i, i])[0], nz[1:]], axis=0)
            G[nz] = (G[nz] + G[i:i+1]) % 2
        return G, True

    @staticmethod
    def gaussjordan(X, change=0):
        """Compute the binary row reduced echelon form of X.
        Parameters
        ----------
        X: array (m, n)
        change : boolean (default, False). If True returns the inverse transform
        Returns
        -------
        if `change` == 'True':
            A: array (m, n). row reduced form of X.
            P: tranformations applied to the identity
        else:
            A: array (m, n). row reduced form of X.
        """
        A = np.copy(X)
        m, n = A.shape

        if change:
            P = np.identity(m).astype(int)

        pivot_old = -1
        for j in range(n):
            filtre_down = A[pivot_old + 1:m, j]
            pivot = np.argmax(filtre_down) + pivot_old + 1

            if A[pivot, j]:
                pivot_old += 1
                if pivot_old != pivot:
                    aux = np.copy(A[pivot, :])
                    A[pivot, :] = A[pivot_old, :]
                    A[pivot_old, :] = aux
                    if change:
                        aux = np.copy(P[pivot, :])
                        P[pivot, :] = P[pivot_old, :]
                        P[pivot_old, :] = aux

                for i in range(m):
                    if i != pivot_old and A[i, j]:
                        if change:
                            P[i, :] = abs(P[i, :] - P[pivot_old, :])
                        A[i, :] = abs(A[i, :] - A[pivot_old, :])

            if pivot_old == m - 1:
                break

        if change:
            return A, P
        return A

    @staticmethod
    def gausselimination(A, b):
        """Solve linear system in Z/2Z via Gauss Gauss elimination."""
        if type(A) == scipy.sparse.csr_matrix:
            A = A.toarray().copy()
        else:
            A = A.copy()
        b = b.copy()
        n, k = A.shape

        for j in range(min(k, n)):
            listedepivots = [i for i in range(j, n) if A[i, j]]
            if len(listedepivots):
                pivot = np.min(listedepivots)
            else:
                continue
            if pivot != j:
                aux = (A[j, :]).copy()
                A[j, :] = A[pivot, :]
                A[pivot, :] = aux

                aux = b[j].copy()
                b[j] = b[pivot]
                b[pivot] = aux

            for i in range(j+1, n):
                if A[i, j]:
                    A[i, :] = abs(A[i, :]-A[j, :])
                    b[i] = abs(b[i]-b[j])

        return A, b

    @staticmethod
    def binaryproduct(X, Y):
        """Compute a matrix-matrix / vector product in Z/2Z."""
        A = X.dot(Y)
        try:
            A = A.toarray()
        except AttributeError:
            pass
        return A % 2
