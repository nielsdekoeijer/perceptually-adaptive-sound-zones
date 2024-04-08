import mosek as mk
import numpy as np

# Generically task formed for problems of the form:
#
#   min  f
#   s.t. ||g|| <= f,
#        ||A_m || <= g_m,       for m in 1 ... M
#        ||B_m - b_m|| <= d_m,  for m in 1 ... M
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
def run_soundzone_mosek_task(task, Ndual, Nw, verbose=False):
    task.optimize()

    # Get dual
    x_dual = Ndual * [0.]
    task.gety(mk.soltype.itr, x_dual)

    solsta = task.getsolsta(mk.soltype.itr)
    if verbose == True:
        print(solsta)

    if solsta != mk.solsta.optimal:
        print(f"Failed to solve with solsta {solsta}... exiting")
        exit()

    # Extract w vector
    w = x_dual[-Nw:]
    return np.array(w)
