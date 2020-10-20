import os
import sys
import numpy as np


class TannerGraph(object):
    def __init__(self, n, m):
        super(TannerGraph, self).__init__()
        self.n = n
        self.m = m
        self.VN = list(range(n))
        self.CN = list(range(n, n + m))
        self.E = set()

    def to_parity_check_matrix(self):
        H = np.zeros((self.n, self.m), dtype=np.int32)
        y = [min(e) for e in self.E]
        x = [max(e) - self.n for e in self.E]
        H[y, x] = 1
        return H

    def add_edge(self, i, j):
        assert i < self.n <= j or j < self.n <= i
        self.E.add({i, j})

    def remove_edge(self, i, j):
        assert i < self.n <= j or j < self.n <= i
        self.E.remove({i, j})

    def has_edge(self, i, j):
        return {i, j} in self.E
