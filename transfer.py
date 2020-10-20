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

from utils.poly_gf2 import poly1d_gf2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('message')
    args = parser.parse_args()

    coder = LDPCCode(36)
    exit()
    # coder = PolynomialCode(4, poly1d_gf2([1, 0, 1, 1]))
    # coder = PolynomialCode(12, poly1d_gf2([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1]))

    """
    for message in coder.iter_blocks():
        code = coder.encode(message)
        for i in range(code.shape[0]):
            code1 = code.copy()
            code1[i] = 1
            message_re = coder.decode(code1)
            print(message, code, code1, message_re)
            if np.any(message != message_re):
                print("Error")
                exit()
    print("Success!")
    exit()
    #"""
    # print(message, code)
    # print(coder.check_has_error(code))
    # print(coder.decode(code))
    # coder = HammingCode(5)
    # coder = IdentityCode(5)

    # channel = BurstErrorChannel(coder, 1)
    # channel = StraightChannel(coder)
    channel = BernoulliChannel(coder)
    # channel = GaussChannel(coder, 330)
    print(channel.transfer_string(args.message))
