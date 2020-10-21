import os
import sys
import argparse
import numpy as np
from codes.qc_ldpc import QuasiCyclicLDPCCode
from utils.poly_gf2 import poly1d_gf2
import importlib


def read_alist(alist):
    """Create H matrix from a file in the .alist format

    See http://www.inference.org.uk/mackay/codes/alist.html for
    documentation on the .alist format

    alist -- string, filename of alist

    returns
    H -- numpy array of size MxN, H matrix
    N -- int, N parameter from alist
    M -- int, M parameter from alist

    """

    with open(alist, 'r') as f:

        line = f.readline()
        N, M = line.split(' ')

        line = f.readline()
        max_col_weight, max_row_weight = line.split(' ')

        line = f.readline()
        col_weights = line.split(' ')
        col_weights.pop()

        line = f.readline()
        row_weights = line.split(' ')
        row_weights.pop()

        nlist = []
        mlist = []

        for i in range(int(N)):
            nlist.append(f.readline().split('\n')[0])

        for i in range(int(M)):
            mlist.append(f.readline().split('\n')[0])

    H = np.zeros((int(M), int(N)), dtype=bool)

    for i in range(int(M)):
        indices = mlist[i].split(' ')[0:int(max_row_weight)]
        indices = [int(x) - 1 for x in indices]

        # print indices
        for k in indices:
            if k != -1:
                H[i][k] = 1
    return H, int(N), int(M)


def C2H_poly_gf2(C, r):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pathToMat')
    args = parser.parse_args()

    dir_path, module_name = os.path.split(args.pathToMat)
    module_name = os.path.splitext(module_name)[0]
    sys.path.append(dir_path)
    data = importlib.import_module(module_name)

    r = data.r
    C = data.C
    print(C.shape)
    H_poly_gf2 = np.empty(C.shape, dtype=poly1d_gf2)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if C[i, j] == -1:
                H_poly_gf2[i, j] = poly1d_gf2([0])
            else:
                H_poly_gf2[i, j] = poly1d_gf2.create_basis(C[i, j])

    dmin_upper_bound = QuasiCyclicLDPCCode.estimate_dmin_upper_bound(H_poly_gf2)
    print(dmin_upper_bound)
