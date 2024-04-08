from pcsz.optimize.run_soundzone_mosek_task import run_soundzone_mosek_task
from pcsz.optimize.generate_soundzone_mosek_task import generate_soundzone_mosek_task

import numpy as np
import cvxpy as cp
import mosek as mk
import pytest

def test_run_soundzone_mosek_task():
    Np = 3
    Nw = 10
    Nb = 20
    Nprim = 1 + Np + 2 * Np * (Nb + 1) + Nw + 1
    Ndual = 1 + Np + Nw
    A = [np.abs(np.random.randn(Nb, Nw)) for m in range(Np)]
    B = [np.abs(np.random.randn(Nb, Nw)) for m in range(Np)]
    b = [np.abs(np.random.randn(Nb)) for m in range(Np)]
    d = [0.8 * np.linalg.norm(b[m]) for m in range(Np)]
    e = 10e7

    with mk.Env() as env:
        with env.Task() as task:
            task = generate_soundzone_mosek_task(task, A, B, b, d, e)
            w_msk = run_soundzone_mosek_task(task, Ndual, Nw)

    w_cvx = cp.Variable(Nw)
    g_cvx = cp.Variable(Np)
    f_cvx = cp.Variable(1)
    p_cvx = cp.Problem(
        cp.Minimize(f_cvx),
        constraints=(
            [cp.SOC(f_cvx, g_cvx)] + 
            [cp.SOC(g_cvx[m], A[m] @ w_cvx) for m in range(Np)] +        
            [cp.SOC(d[m], B[m] @ w_cvx - b[m]) for m in range(Np)]        
    ))
    p_cvx.solve(verbose=False)
    assert all(np.abs(w_cvx.value - w_msk) < 1e-5)

if __name__=="__main__":
    test_run_soundzone_mosek_task()