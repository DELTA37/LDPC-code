from .linear import LinearCode
import numpy as np


class CyclicCode(LinearCode):
    def __init__(self):
        super(CyclicCode, self).__init__()

    def encode(self, array: np.ndarray) -> np.ndarray:
        pass

    def decode(self, array: np.ndarray) -> np.ndarray:
        pass
