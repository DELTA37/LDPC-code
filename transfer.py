import argparse
import numpy as np
from noise_channel.straight import StraightChannel
from noise_channel.bernoulli import BernoulliChannel
from noise_channel.gaussian import GaussChannel

from codes.hamming import HammingCode
from codes.identity import IdentityCode
from codes.polynomial import PolynomialCode


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('message')
    args = parser.parse_args()
    # coder = PolynomialCode(3, np.array([1, 1, 1]))
    coder = HammingCode(5)
    # coder = IdentityCode(5)
    # channel = StraightChannel(coder)
    # channel = BernoulliChannel(coder)
    channel = GaussChannel(coder, 330)
    print(channel.transfer_string(args.message))
