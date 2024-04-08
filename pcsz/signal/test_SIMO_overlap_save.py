from pcsz.signal.SIMO_overlap_save import SIMO_overlap_save

import numpy as np
import scipy as sp
import pytest

if __name__ == "__main__":
    import matplotlib.pyplot as plt

import scipy.signal

def test_SIMO_overlap_save_part():
    Nx = 800
    Nb = 80
    Nw = 40
    Nr = 20
    Nt = 40
    Nl = 20

    x = np.random.randn(Nx)
    w = [np.random.randn(Nw) for l in range(Nl)]
    y = [np.convolve(x, w[l]) for l in range(Nl)]

    window = sp.signal.windows.hann(Nt, sym=False)
    overlap_save = SIMO_overlap_save(x, Nb, Nr, Nt, Nw, window, Nl)
    
    # Test to see if input blocks work
    input_test_block_0 = overlap_save.get_input_block(0)
    assert input_test_block_0.shape[0] == Nb
    assert all(input_test_block_0 == np.pad(x[0 : 1 * Nr], (Nb - 1 * Nr, 0)))

    input_test_block_1 = overlap_save.get_input_block(1)
    assert input_test_block_1.shape[0] == Nb
    assert all(input_test_block_1 == np.pad(x[0 : 2 * Nr], (Nb - 2 * Nr, 0)))
    
    input_test_block_2 = overlap_save.get_input_block(2)
    assert input_test_block_2.shape[0] == Nb
    assert all(input_test_block_2 == np.pad(x[0 : 3 * Nr], (Nb - 3 * Nr, 0)))
    
    # Test to see if output blocks are valid
    output_test_block_0 = overlap_save.set_output_block(0, w)
    for l in range(Nl):
        assert output_test_block_0[l].shape[0] == Nt
        assert all(abs(y[l][0 : Nr] - output_test_block_0[l][-Nr:]) < 1e-10)

    output_test_block_1 = overlap_save.set_output_block(1, w)
    for l in range(Nl):
        assert output_test_block_1[l].shape[0] == Nt
        assert all(abs(y[l][0 : Nt] - output_test_block_1[l]) < 1e-10)

    output_test_block_2 = overlap_save.set_output_block(2, w)
    for l in range(Nl):
        assert output_test_block_2[l].shape[0] == Nt
        assert all(abs(y[l][Nr : Nt + Nr] - output_test_block_2[l]) < 1e-10)
    
def test_SIMO_overlap_save_full():
    Nx = 800
    Nb = 80
    Nw = 40
    Nr = 20
    Nt = 40
    Nl = 20

    x = np.random.randn(Nx)
    w = [np.random.randn(Nw) for l in range(Nl)]
    y = [np.convolve(x, w[l]) for l in range(Nl)]

    window = sp.signal.windows.hann(Nt, sym=False)
    overlap_save = SIMO_overlap_save(x, Nb, Nr, Nt, Nw, window, Nl)

    for s in range(overlap_save.smin, overlap_save.smax + 1):
        overlap_save.set_output_block(s, w)

    for l in range(Nl):
        assert all(abs(y[l] - overlap_save.y[l][0 : Nx + Nw - 1]) < 1e-10)

    if __name__ == "__main__":
        plt.plot(y[0])
        plt.plot(overlap_save.y[0])
        plt.show()
    
if __name__=="__main__":
    test_SIMO_overlap_save_part()
    test_SIMO_overlap_save_full()
