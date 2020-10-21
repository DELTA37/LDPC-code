from .base import BaseCode
import numpy as np
from typing import Tuple
import scipy
import scipy.sparse


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
        return self.binaryproduct(array, self.G)

    def decode(self, array: np.ndarray) -> np.ndarray:
        # TODO: table of error patterns
        raise NotImplementedError()

    def check_has_error(self, array: np.ndarray):
        return not (self.binaryproduct(self.H, array) == 0).all()

    @staticmethod
    def coding_matrix_systematic(H, sparse=True):
        """Compute a coding matrix G in systematic format with an identity block.
        Parameters
        ----------
        H: array (n_equations, n_code). Parity-check matrix.
        sparse: (boolean, default True): if `True`, scipy.sparse is used
        to speed up computation if n_code > 1000.
        Returns
        -------
        H_new: (n_equations, n_code) array. Modified parity-check matrix given by a
            permutation of the columns of the provided H.
        G_systematic.T: Transposed Systematic Coding matrix associated to H_new.
        """
        n_equations, n_code = H.shape

        if n_code > 1000 or sparse:
            sparse = True
        else:
            sparse = False

        P1 = np.identity(n_code, dtype=int)

        Hrowreduced = LinearCode.gaussjordan(H)

        n_bits = n_code - sum([a.any() for a in Hrowreduced])

        # After this loop, Hrowreduced will have the form H_ss : | I_(n-k)  A |

        while True:
            zeros = [i for i in range(min(n_equations, n_code)) if not Hrowreduced[i, i]]
            if len(zeros):
                indice_colonne_a = min(zeros)
            else:
                break
            list_ones = [j for j in range(indice_colonne_a + 1, n_code) if Hrowreduced[indice_colonne_a, j]]
            if len(list_ones):
                indice_colonne_b = min(list_ones)
            else:
                break
            aux = Hrowreduced[:, indice_colonne_a].copy()
            Hrowreduced[:, indice_colonne_a] = Hrowreduced[:, indice_colonne_b]
            Hrowreduced[:, indice_colonne_b] = aux

            aux = P1[:, indice_colonne_a].copy()
            P1[:, indice_colonne_a] = P1[:, indice_colonne_b]
            P1[:, indice_colonne_b] = aux

        # Now, Hrowreduced has the form: | I_(n-k)  A | ,
        # the permutation above makes it look like :
        # |A  I_(n-k)|

        P1 = P1.T
        identity = list(range(n_code))
        sigma = identity[n_code - n_bits:] + identity[:n_code - n_bits]

        P2 = np.zeros(shape=(n_code, n_code), dtype=int)
        P2[identity, sigma] = np.ones(n_code)

        if sparse:
            P1 = scipy.sparse.csr_matrix(P1)
            P2 = scipy.sparse.csr_matrix(P2)
            H = scipy.sparse.csr_matrix(H)

        P = LinearCode.binaryproduct(P2, P1)

        if sparse:
            P = scipy.sparse.csr_matrix(P)

        H_new = LinearCode.binaryproduct(H, np.transpose(P))

        G_systematic = np.zeros((n_bits, n_code), dtype=int)
        G_systematic[:, :n_bits] = np.identity(n_bits)
        G_systematic[:, n_bits:] = (Hrowreduced[:n_code - n_bits, n_code - n_bits:]).T
        return H_new, G_systematic

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

    @staticmethod
    def get_bits_and_nodes(H):
        """Return bits and nodes of a parity-check matrix H."""
        if type(H) != scipy.sparse.csr_matrix:
            bits_indices, bits = np.where(H)
            nodes_indices, nodes = np.where(H.T)
        else:
            bits_indices, bits = scipy.sparse.find(H)[:2]
            nodes_indices, nodes = scipy.sparse.find(H.T)[:2]
        bits_histogram = np.bincount(bits_indices)
        nodes_histogram = np.bincount(nodes_indices)

        return bits_histogram, bits, nodes_histogram, nodes
