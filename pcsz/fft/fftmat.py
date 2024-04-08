import numpy as np
import scipy as sp

def srfftmat(N):
    if N % 2 != 0: raise "Uneven srfft matrix not currently support."
    return np.vstack([
                np.real(np.fft.fft(np.eye(N))[0 : N // 2 + 1, :]), 
                np.imag(np.fft.fft(np.eye(N))[1 : N // 2, :])
            ])

def isrfftmat(N):
    if N % 2 != 0: raise "Uneven isrfft matrix not currently support."
    return (1 / N) * np.hstack([
            np.real(np.fft.fft(np.eye(N))[0 : N // 2 + 1, :].T), 
            np.imag(np.fft.fft(np.eye(N))[1 : N // 2, :].T)
        ]) @ np.diag(
                [1.] + (N // 2 - 1) * [2.] + [1.] + (N // 2 - 1) * [2.]
        ) 

def iminimal_real_vector(x, N):
    if x.ndim == 1:
        if N % 2 == 0:
            return x[0 : (x.size + 2) // 2] + 1j * np.pad(x[(x.size + 2) // 2 :], (1,1))
        else:
            return x[0 : (x.size + 1) // 2] + 1j * np.pad(x[(x.size + 1) // 2 :], (1,0))
    elif x.ndim == 2:
        if N % 2 == 0:
            return x[0 : (x.shape[0] + 2) // 2, :] + 1j * np.pad(x[(x.shape[0] + 2) // 2 :, :], ((1,1), (0,0)))
        else:
            return x[0 : (x.shape[0] + 1) // 2, :] + 1j * np.pad(x[(x.shape[0] + 1) // 2 :, :], ((1,0), (0,0)))
    else:
        raise f"Dimension {x.ndim} unsupported."

def minimal_real_vector(x, N):
    if x.ndim == 1:
        if N % 2 == 0:
            return np.block([np.real(x), np.imag(x)[1 : x.size - 1]])
        else:
            return np.block([np.real(x), np.imag(x)[1 : x.size]])
    elif x.ndim == 2:
        if N % 2 == 0:
            return np.block([[np.real(x)], [np.imag(x)[1 : x.shape[0] - 1, :]]])
        else:
            return np.block([[np.real(x)], [np.imag(x)[1 : x.shape[0] , :]]])
    else:
        raise f"Dimension {x.ndim} unsupported."

def minimal_real_matrix(X):
    return sp.sparse.bmat([
        [np.real(X), -np.imag(X)[:, 1 : X.shape[1] - 1]],
        [np.imag(X)[1 : X.shape[0] - 1, :], np.real(X)[1 : X.shape[0] - 1, 1 : X.shape[1] - 1]]
    ])
