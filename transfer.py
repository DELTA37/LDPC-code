import argparse
import numpy as np

from noise_channel.straight import StraightChannel
from noise_channel.bernoulli import BernoulliChannel
from noise_channel.gaussian import GaussChannel
from noise_channel.burst_error import BurstErrorChannel

from codes.hamming import HammingCode
from codes.identity import IdentityCode
from codes.polynomial import PolynomialCode
from codes.poly_gf2 import poly1d_gf2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('message')
    args = parser.parse_args()
    coder = PolynomialCode(4, poly1d_gf2([1, 0, 1, 1]))
    # message = np.array([0, 1, 0, 0])
    # code = coder.encode(message)
    # print(message, code)
    # print(coder.check_has_error(code))
    # print(coder.decode(code))
    # coder = HammingCode(5)
    # coder = IdentityCode(5)

    channel = BurstErrorChannel(coder, 3)
    # channel = StraightChannel(coder)
    # channel = BernoulliChannel(coder)
    # channel = GaussChannel(coder, 330)
    print(channel.transfer_string(args.message))
