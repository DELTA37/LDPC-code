import os
import sys
import numpy as np


def find_smallest(array):
    if len(array) == 1:
        return 0
    elif len(array) == 2:
        if array[0] <= array[1]:
            return 0
        else:
            return 1
    else:
        arrayA = array[:len(array)/2]
        arrayB = array[(len(array)/2):]
        smallA = find_smallest(arrayA)
        smallB = find_smallest(arrayB)
        if arrayA[smallA] <= arrayB[smallB]:
            return smallA
        else:
            return len(arrayA) + smallB


class TannerGraph(object):
    """
    tested with https://uzum.github.io/ldpc-peg/
    """
    def __init__(self, n, m):
        super(TannerGraph, self).__init__()
        self.n = n  # var nodes
        self.m = m  # check nodes
        assert n >= m
        self.H = np.zeros((self.m, self.n), dtype=np.int32)

    def to_parity_check_matrix(self):
        return self.H

    def add_edge(self, v, c):
        self.H[c, v] = 1

    def remove_edge(self, v, c):
        self.H[c, v] = 0

    def has_edge(self, v, c):
        return self.H[c, v]

    @staticmethod
    def create_with_PEG(n, m, vn_degrees):
        """
        Progressive Edge Growth Algorithm in Tanner Graphs
        :param vn_degrees:
        :return:
        """
        assert vn_degrees.shape[0] == n
        # vn_degrees = np.sort(vn_degrees)

        t = TannerGraph(n, m)

        for i in range(n):
            for k in range(vn_degrees[i]):
                if k == 0:
                    c_min_degree = np.argmin(np.sum(t.H, axis=-1))
                    t.H[c_min_degree, i] = 1
                else:
                    vn_ids = [i]
                    ch_ids = []

                    ch_degrees = np.zeros(m, dtype=np.int32)
                    vn_not_visited = np.ones(n, dtype=np.int32)
                    ch_not_visited = np.ones(m, dtype=np.int32)

                    vn_not_visited[vn_ids] = 0
                    ch_not_visited[ch_ids] = 0
                    while len(vn_ids) > 0 or len(ch_ids) > 0:
                        if len(ch_ids) > 0:
                            ch = ch_ids.pop(0)
                            assert not ch_not_visited[ch]

                            new_vn_ids: list = np.where(t.H[ch] * vn_not_visited)[0].tolist()
                            vn_not_visited[new_vn_ids] = 0
                            vn_ids.extend(new_vn_ids)
                            ch_degrees[ch] = 1 + len(new_vn_ids)

                        elif len(vn_ids) > 0:
                            vn = vn_ids.pop(0)
                            assert not vn_not_visited[vn]

                            new_ch_ids = np.where(t.H[:, vn] * ch_not_visited)[0]
                            ch_not_visited[new_ch_ids] = 0
                            ch_ids.extend(new_ch_ids)
                    ch_degrees[np.where(t.H[:, i])[0]] = n + 1
                    c_min_degree = np.argmin(ch_degrees)
                    t.add_edge(i, c_min_degree)
        return t
