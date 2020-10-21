from .bp_code import BPCode
import numpy as np
from itertools import combinations
from utils.poly_gf2 import poly1d_gf2
from permanent.permanent import permanent
from tqdm import tqdm


class QuasiCyclicLDPCCode(BPCode):
    def __init__(self, r, J, I,
                 snr=20,
                 maxiter=100):

        self.r = r
        self.I = I
        self.J = J
        code_size = r * I
        n_equations = r * J
        self.H_poly_gf2 = None

        H = None
        H, G = self.coding_matrix_systematic(H)

        super(QuasiCyclicLDPCCode, self).__init__(block_size=G.shape[0], code_size=code_size,
                                                  G=G, H=H,
                                                  snr=snr,
                                                  maxiter=maxiter)

    def create_parity_check_matrix(self, code_size: int, d_v: int, d_c: int):
        # TODO
        raise NotImplementedError()

    @staticmethod
    def estimate_dmin_upper_bound(H_poly_gf2):
        """
        Theorem 8 from https://arxiv.org/pdf/0901.4129.pdf
        :param H_poly_gf2:
        :return:
        """

        def permanent_int(A):
            return int(permanent(A.astype(np.complex)).real)

        J, I = H_poly_gf2.shape
        A = np.array([[H_poly_gf2[y, x].weight() for x in range(I)] for y in range(J)])
        dmin_upper_bound = min([sum([permanent_int(A[:, ids[:i] + ids[i + 1:]])
                                     for i in range(J + 1)])
                               for ids in tqdm(list(combinations(range(I), J + 1)))])
        return dmin_upper_bound
