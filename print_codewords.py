import argparse
import numpy as np
from noise_channel.straight import StraightChannel
from noise_channel.bernoulli import BernoulliChannel
from codes.hamming import HammingCode
from codes.identity import IdentityCode
from codes.polynomial import PolynomialCode


if __name__ == '__main__':
    coder = PolynomialCode(3, np.array([1, 1, 1]))

    for block, code in coder.iter_codewords():
        print(f"{block} -> {code}")
