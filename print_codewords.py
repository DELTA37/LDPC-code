import argparse
import numpy as np
from noise_channel.straight import StraightChannel
from noise_channel.bernoulli import BernoulliChannel
from codes.hamming import HammingCode
from codes.identity import IdentityCode
from codes.polynomial import PolynomialCode
from codes.ldpc import LDPCCode


if __name__ == '__main__':
    # coder = HammingCode(3)
    coder = LDPCCode(20, 4, 5)
    # coder = PolynomialCode(3, np.array([1, 1, 1]))

    for block, code in coder.iter_blocks_codewords():
        print(f"{block} -> {code}")

    dmin = coder.calc_dmin()
    print(f"dmin: {dmin}")
    print(f"max error detect: {dmin - 1}")
    print(f"max error correct: {(dmin - 1) // 2}")
