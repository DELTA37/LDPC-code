import os
from .bp_code import BPCode
from utils.log_bp_solver import _logbp_numba, _logbp_numba_regular
import numpy as np
import scipy
import scipy.sparse


class LDPCCode(BPCode):
    def __init__(self, code_size,
                 d_v=2, d_c=4,
                 snr=20,
                 maxiter=100):
        H = self.create_parity_check_matrix(code_size, d_v, d_c)
        H, G = self.coding_matrix_systematic(H)

        self.d_v = d_v
        self.d_c = d_c
        super(LDPCCode, self).__init__(block_size=G.shape[0], code_size=code_size,
                                       G=G, H=H,
                                       snr=snr,
                                       maxiter=maxiter)

    def create_parity_check_matrix(self, code_size: int, d_v: int, d_c: int):
        """
        2.1 Gallager ensemble
        (Low-Density Parity-CheckCode Constructions
        Enrico Paolini* and Mark Flanagan†*
        Department of Electrical, Electronic and Information Engineering“G. Marconi”,
        University of Bologna,
        Via Venezia 52, Cesena, FC 47521,
        Italy†School of Electrical, Electronic and Communications Engineering,
        University College Dublin, Belfield, Dublin 4, Ireland)

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
