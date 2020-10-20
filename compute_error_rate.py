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
    parser.add_argument('snr', type=float)
    parser.add_argument('channel_snr', type=float)
    args = parser.parse_args()

    snr = args.snr
    channel_snr = args.channel_snr
    coder = LDPCCode(15, 4, 5, snr=snr)
    # coder = PolynomialCode(4, poly1d_gf2([1, 0, 1, 1]))
    # coder = PolynomialCode(12, poly1d_gf2([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1]))

    err_num = 0
    count = 0
    for message in coder.iter_blocks():
        code = (-1) ** coder.encode(message) + np.random.randn(coder.code_size) * (10 ** (-channel_snr / 20))
        message_re = coder.decode(code)
        if np.any(message != message_re):
            print(message, code, message_re)
            err_num += 1
        count += 1

    print(f"Error rate: {err_num / count}")
