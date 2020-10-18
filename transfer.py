import argparse
from noise_channel.straight import StraightChannel
from noise_channel.bernoulli import BernoulliChannel
from codes.hamming import HammingCode


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('message')
    args = parser.parse_args()
    # channel = StraightChannel(HammingCode(3))
    channel = BernoulliChannel(HammingCode(3))
    print(channel.transfer(args.message))
