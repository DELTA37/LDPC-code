import os
import sys
import numpy as np
from utils.tanner import TannerGraph


if __name__ == '__main__':
    t = TannerGraph.create_with_PEG(8, 4, np.array([2, 3, 2, 2, 2, 4, 2, 2]))
    print(t.H)
    """
    the same output as https://uzum.github.io/ldpc-peg/
    """
