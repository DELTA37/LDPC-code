import os
from .linear import LinearCode
from utils.log_bp_solver import _logbp_numba, _logbp_numba_regular
import numpy as np
import scipy
import scipy.sparse


class BPCode(LinearCode):
    """
    code with belief prop decoder
    """
    def __init__(self, block_size, code_size,
                 G=None, H=None,
                 snr=20, maxiter=100):
        self.snr = snr
        self.maxiter = maxiter
        super(BPCode, self).__init__(block_size=block_size, code_size=code_size,
                                     G=G, H=H)

    def decode(self, array: np.ndarray) -> np.ndarray:
        bits_hist, bits_values, nodes_hist, nodes_values = self.get_bits_and_nodes(self.H)
        _n_bits = np.unique(self.H.sum(0))
        _n_nodes = np.unique(self.H.sum(1))

        if _n_bits.shape[0] == 1 and _n_nodes.shape[0] == 1 and _n_bits * _n_nodes == 1:
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
