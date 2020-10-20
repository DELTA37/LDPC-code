import os
from .linear import LinearCode
from utils.log_bp_solver import _logbp_numba, _logbp_numba_regular
import numpy as np
import scipy
import scipy.sparse


class LDPCCode(LinearCode):
    def __init__(self, code_size,
                 d_v=2, d_c=4,
                 snr=20,
                 maxiter=100):
        H = self.create_parity_check_matrix(code_size, d_v, d_c)
        H, G = self.coding_matrix_systematic(H)

        self.d_v = d_v
        self.d_c = d_c
        self.snr = snr
        self.maxiter = maxiter
        print(G.shape, H.shape)
        super(LDPCCode, self).__init__(block_size=G.shape[0], code_size=code_size,
                                       H=H, G=G)

    @staticmethod
    def create_parity_check_matrix(code_size: int, d_v: int, d_c: int):
        """
        Callager algo
        :param code_size:
        :param d_v:
        :param d_c:
        :return:
        """
        assert d_v > 1
        assert d_c > d_v
        assert code_size % d_c == 0

        n_equations = (code_size * d_v) // d_c
        block = np.zeros((n_equations // d_v, code_size), dtype=int)
        H = np.empty((n_equations, code_size), dtype=np.int32)
        block_size = n_equations // d_v

        for i in range(block_size):
            for j in range(i * d_c, (i + 1) * d_c):
                block[i, j] = 1
        H[:block_size] = block

        for i in range(1, d_v):
            H[i * block_size: (i + 1) * block_size] = np.random.permutation(block.T).T
        return H

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

        Hrowreduced = LDPCCode.gaussjordan(H)

        n_bits = n_code - sum([a.any() for a in Hrowreduced])

        # After this loop, Hrowreduced will have the form H_ss : | I_(n-k)  A |

        while True:
            zeros = [i for i in range(min(n_equations, n_code))
                     if not Hrowreduced[i, i]]
            if len(zeros):
                indice_colonne_a = min(zeros)
            else:
                break
            list_ones = [j for j in range(indice_colonne_a + 1, n_code)
                         if Hrowreduced[indice_colonne_a, j]]
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

        P = LDPCCode.binaryproduct(P2, P1)

        if sparse:
            P = scipy.sparse.csr_matrix(P)

        H_new = LDPCCode.binaryproduct(H, np.transpose(P))

        G_systematic = np.zeros((n_bits, n_code), dtype=int)
        G_systematic[:, :n_bits] = np.identity(n_bits)
        G_systematic[:, n_bits:] = (Hrowreduced[:n_code - n_bits, n_code - n_bits:]).T
        return H_new, G_systematic

    def decode(self, array: np.ndarray) -> np.ndarray:
        bits_hist, bits_values, nodes_hist, nodes_values = self.get_bits_and_nodes(self.H)
        _n_bits = np.unique(self.H.sum(0))
        _n_nodes = np.unique(self.H.sum(1))

        if _n_bits * _n_nodes == 1:
            solver = _logbp_numba_regular
            bits_values = bits_values.reshape(self.code_size, -1)
            nodes_values = nodes_values.reshape(self.H.shape[0], -1)
        else:
            solver = _logbp_numba

        var = 10 ** (-self.snr / 10)

        array = array[:, None]
        Lc = 2 * array / var
        Lq = np.zeros(shape=(self.H.shape[0], self.code_size, 1))
        Lr = np.zeros(shape=(self.H.shape[0], self.code_size, 1))
        for n_iter in range(self.maxiter):
            Lq, Lr, L_posteriori = solver(bits_hist, bits_values, nodes_hist,
                                          nodes_values, Lc, Lq, Lr, n_iter)
            x = np.array(L_posteriori <= 0).astype(np.int32)
            if not self.check_has_error(x):
                break
        return x.squeeze()[:self.block_size]
