import mosek as mk
import scipy as sp
import numpy as np

import scipy.sparse

import sys
def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()

# Generically creates tasks for problems of the form:
#
#   min  f
#   s.t. ||g|| <= f,
#        ||A_m @ w || <= g_m,       for m in 1 ... M
#        ||B_m @ w - b_m|| <= d_m,  for m in 1 ... M
#        ||w|| <= e
#
# By converting to:
# 
#   min  c.T @ x
#   s.t. Ax + b = s
#        s in K
#
# Which has the corresponding dual:
# 
#   max  -b.T @ y
#   s.t. A.T @ y = c
#        y in K^*
#
def generate_soundzone_mosek_task(task, A, B, b, d, e, verbose=False):
    task.putintparam(mk.iparam.presolve_use, mk.presolvemode.off)
    if verbose == True:
        task.set_Stream(mk.streamtype.log, streamprinter)


    # Define inf to be zero for syntaxic sugar
    inf = 0

    # Problem parameters
    N_a = len(A)
    N_b = len(B)

    # A and B assumed same shape...!
    N_c = A[0].shape[0]
    N_w = A[0].shape[1]

    # Define primal and dual variable sizes
    N_dual = 1 + N_a + N_w
    N_prim = 1 + N_a + (N_a + N_b) * (N_c + 1) + (N_w + 1)

    # Generate c vector
    c_msk = sp.sparse.csr_matrix(
            [1.] + (N_w + N_a) * [0.],
        )
    assert c_msk.shape[1] == N_dual

    # Generate b vector
    b_msk = sp.sparse.hstack([
            # gamma cone
            sp.sparse.csr_matrix(
                (1 + N_a) * [0.],
            ),
            # A cone
            sp.sparse.csr_matrix(
                (N_a * (N_c + 1)) * [0.],
            ),
            # B cone
            sp.sparse.hstack(
                [
                    sp.sparse.hstack([d[m]] + (-1 * b[m]).tolist())
                    for m in range(N_b)
                ]),
            # E cone
            sp.sparse.hstack([e] + N_w * [0.])
        ], format="csr")

    assert b_msk.shape[1] == N_prim;

    # Generate A matrix
    A_msk = sp.sparse.vstack([
            # gamma cone
            sp.sparse.eye(N_a + 1, N_dual),
            # A cone
            sp.sparse.vstack([
                sp.sparse.vstack([
                    sp.sparse.coo_matrix(
                        ([1], ([0], [m + 1])), 
                        shape=(1, N_dual)
                    ),
                    sp.sparse.hstack([
                        sp.sparse.coo_matrix(np.zeros((N_c, N_a + 1))),
                        A[m]
                    ])
                ])
                for m in range(N_a)
            ]),
            # B cone
            sp.sparse.vstack([
                sp.sparse.vstack([
                    sp.sparse.coo_matrix(
                        ([0], ([0], [0])), 
                        shape=(1, N_dual)
                    ),
                    sp.sparse.hstack([
                        sp.sparse.coo_matrix(np.zeros((N_c, N_a + 1))),
                        B[m]
                    ])
                ])
                for m in range(N_b)
            ]),
            # E cone
            sp.sparse.vstack([
                sp.sparse.coo_matrix(
                    ([0], ([0], [0])), 
                    shape=(1, N_dual)
                ),
                sp.sparse.hstack([
                    sp.sparse.coo_matrix(np.zeros((N_w, N_a + 1))),
                    sp.sparse.eye(N_w)
                ])
            ])
        ]).T
    assert A_msk.shape == (N_dual, N_prim)

    # Define variables
    bkx = N_prim * [mk.boundkey.fr]
    blx = N_prim * [-inf]
    bux = N_prim * [ inf]

    task.appendvars(N_prim)
    task.putvarboundlist([j for j in range(N_prim)], bkx, blx, bux)

    # Define linear constraints
    bkc = N_dual * [mk.boundkey.fx]
    assert len(bkc) == N_dual
    blc = c_msk.toarray()[0]
    assert len(blc) == N_dual
    buc = c_msk.toarray()[0]
    assert len(buc) == N_dual
    A_msk_r, A_msk_c, A_msk_v = sp.sparse.find(A_msk)

    task.appendcons(N_dual)
    task.putaijlist(A_msk_r.tolist(), A_msk_c.tolist(), A_msk_v.tolist())
    task.putconboundlist([j for j in range(N_dual)], bkc, blc, buc)

    # Define conic constraints
    task.appendconesseq(
            (1 + (N_a + N_b) + 1) * [mk.conetype.quad], 
            (1 + (N_a + N_b) + 1) * [0.], 
            [N_a + 1] + (N_a + N_b) * [N_c + 1] + [N_w + 1], 
            0
        )

    # Define cost
    for j in range(N_prim):
        task.putcj(j, -1 * b_msk[0, j])
    task.putobjsense(mk.objsense.maximize)

    return task
