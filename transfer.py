import argparse
import numpy as np

from noise_channel.straight import StraightChannel
from noise_channel.bernoulli import BernoulliChannel
from noise_channel.gaussian import GaussChannel
from noise_channel.burst_error import BurstErrorChannel

from codes.hamming import HammingCode
from codes.identity import IdentityCode
from codes.polynomial import PolynomialCode
from codes.ldpc import LDPCCode
from codes.peg_ldpc import PEGLDPCCode

from utils.poly_gf2 import poly1d_gf2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('message')
    args = parser.parse_args()

    coder = PEGLDPCCode(4, 8, np.array([1, 2, 1, 1, 1, 2, 2, 2]))
    # coder = LDPCCode(1005, 4, 5)
    # coder = PolynomialCode(4, poly1d_gf2([1, 0, 1, 1]))
    # coder = PolynomialCode(12, poly1d_gf2([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1]))
    # exit()
    #"""
    # print(message, code)
    # print(coder.check_has_error(code))
    # print(coder.decode(code))
    # coder = HammingCode(5)
    # coder = IdentityCode(5)

    # channel = BurstErrorChannel(coder, 1)
    # channel = StraightChannel(coder)
    # channel = BernoulliChannel(coder)
    channel = GaussChannel(coder, coder.snr)
    print(channel.transfer_string(args.message))
