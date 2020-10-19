from codes.linear import LinearCode
import numpy as np


if __name__ == '__main__':
    G = np.zeros((3, 11), dtype=np.int32)
    G[0, 1] = 1
    G[2, 10] = 1
    G[1, [3, 7, 8]] = 1
    G[:, -1] = 1
    print(G)
    G, res = LinearCode.bring_matrix_to_identity_residual_form(G)
    print(G)
    print(res)
