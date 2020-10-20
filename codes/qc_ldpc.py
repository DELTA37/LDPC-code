from .ldpc import LDPCCode


class QuasiCyclicLDPCCode(LDPCCode):
    def __init__(self, code_size,
                 d_v=2, d_c=4,
                 snr=20,
                 maxiter=100):
        super(QuasiCyclicLDPCCode, self).__init__(code_size=code_size,
                                                  d_v=d_v,
                                                  d_c=d_c, snr=snr,
                                                  maxiter=maxiter)

    def create_parity_check_matrix(self, code_size: int, d_v: int, d_c: int):
        raise NotImplementedError()
