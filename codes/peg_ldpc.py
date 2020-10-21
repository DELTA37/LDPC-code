from .bp_code import BPCode
import numpy as np
from utils.tanner import TannerGraph


class PEGLDPCCode(BPCode):
    def __init__(self, block_size, code_size,
                 vn_degrees,
                 snr=20,
                 maxiter=100):
        H = self.create_parity_check_matrix(block_size, code_size, vn_degrees)
        H, G = self.coding_matrix_systematic(H)

        super(PEGLDPCCode, self).__init__(block_size=block_size, code_size=code_size,
                                          G=G, H=H,
                                          snr=snr,
                                          maxiter=maxiter)

    @staticmethod
    def create_parity_check_matrix(block_size, code_size, vn_degrees):
        t = TannerGraph.create_with_PEG(code_size, code_size - block_size, vn_degrees)
        return t.H
