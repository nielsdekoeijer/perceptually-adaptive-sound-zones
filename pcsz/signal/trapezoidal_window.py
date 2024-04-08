import numpy as np
import scipy as sp
import scipy.signal

def trapezoidal_window(Nb, Nr):
    window = np.concatenate([
        np.linspace(0, 1, Nb // 8), 
        np.ones(Nb - 2 * (Nb // 8)), 
        np.linspace(1, 0, Nb // 8)
    ])

    assert sp.signal.check_COLA(window, Nb, Nr), f"COLA check not passed for window of size {Nb} and hopsize of {Nr}"

    return window