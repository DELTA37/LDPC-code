import os
import numpy as np
import math
import bitarray


class BaseCode(object):
    def __init__(self):
        super(BaseCode, self).__init__()

    def encode(self, array: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def decode(self, array: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def is_ascii(s):
        return all(ord(c) < 128 for c in s)

    @staticmethod
    def string2bit(s: str) -> np.ndarray:
        assert BaseCode.is_ascii(s)
        x = bitarray.bitarray(''.join([bin(ord(x))[2:].zfill(8) for x in s]))
        return np.array(x.tolist(), dtype=np.int32)

    @staticmethod
    def bit2string(x: np.ndarray) -> str:
        assert x.shape[0] % 8 == 0
        x = ''.join(map(str, (x % 2).tolist()))
        x = ''.join([chr(int(x[i: i + 8], 2)) for i in range(0, len(x), 8)])
        return x

    @staticmethod
    def matmul(a: np.ndarray, x: np.ndarray) -> np.ndarray:
        return a.dot(x) % 2

    @staticmethod
    def flip(x: np.ndarray) -> np.ndarray:
        return (x + 1) % 2
