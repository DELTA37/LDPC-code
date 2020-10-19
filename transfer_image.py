import argparse
import numpy as np
from PIL import Image
from noise_channel.straight import StraightChannel
from noise_channel.bernoulli import BernoulliChannel
from noise_channel.gaussian import GaussChannel

from codes.hamming import HammingCode
from codes.identity import IdentityCode
from codes.polynomial import PolynomialCode


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('output')
    args = parser.parse_args()
    # coder = PolynomialCode(3, np.array([1, 1, 1]))
    coder = HammingCode(5)
    # coder = IdentityCode(5)
    # channel = StraightChannel(coder)
    # channel = BernoulliChannel(coder)
    channel = GaussChannel(coder, 330)
    image = channel.transfer_image(Image.open(args.image))
    image.save(args.output)
