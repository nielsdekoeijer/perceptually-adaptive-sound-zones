from pcsz.fft.fftmat import minimal_real_vector, iminimal_real_vector, minimal_real_matrix, srfftmat, isrfftmat

import pytest
import numpy as np
import scipy as sp
import scipy.sparse

def test_srfft():
    x = np.random.randn(10)
    F = srfftmat(10)
    B = isrfftmat(10)
    assert np.linalg.norm(x - B @ F @ x) < 1e-8

def test_minimal_vector():
    # Even 1D test
    x = np.random.randn(10)
    y = minimal_real_vector(np.fft.rfft(x), 10)
    assert np.linalg.norm(x - np.fft.irfft(iminimal_real_vector(y, 10))) < 1e-8

    # Uneven 1D test
    x = np.random.randn(11)
    y = minimal_real_vector(np.fft.rfft(x), 11)
    assert np.linalg.norm(x - np.fft.irfft(iminimal_real_vector(y, 11), n=11)) < 1e-8

    # Even 2D test
    x = np.array([np.random.randn(10) for m in range(5)]).T
    y = minimal_real_vector(np.fft.rfft(x, axis=0), 10)
    assert np.linalg.norm(x - np.fft.irfft(iminimal_real_vector(y, 10), axis=0, n=10)) < 1e-8

    # Uneven 2D test
    x = np.array([np.random.randn(11) for m in range(5)]).T
    y = minimal_real_vector(np.fft.rfft(x, axis=0), 11)
    assert np.linalg.norm(x - np.fft.irfft(iminimal_real_vector(y, 11), axis=0, n=11)) < 1e-8

def test_minimal_matrix():
    # Even matrix tests
    A = sp.sparse.csr_matrix(np.diag(np.fft.rfft(np.random.randn(10))))
    x = np.fft.rfft(np.random.randn(10))
    assert np.linalg.norm(A @ x - iminimal_real_vector(minimal_real_matrix(A) @ minimal_real_vector(x, 6), 10)) < 1e-8


if __name__=="__main__":
    test_srfft()
    test_minimal_vector()
    test_minimal_matrix()