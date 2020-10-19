import argparse
import numpy as np
from noise_channel.straight import StraightChannel
from noise_channel.bernoulli import BernoulliChannel
from codes.hamming import HammingCode
from codes.identity import IdentityCode


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('message')
    args = parser.parse_args()
    coder = HammingCode(5)
    # coder = IdentityCode(5)
    # channel = StraightChannel(coder)
    channel = BernoulliChannel(coder)
    print(channel.transfer(args.message))
