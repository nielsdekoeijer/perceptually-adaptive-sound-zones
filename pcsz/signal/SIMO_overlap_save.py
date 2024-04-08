import numpy as np
import scipy as sp


import scipy as sp

class SIMO_overlap_save:
    def __init__(self, x, Nb : int, Nr : int, Nt : int, Nw : int, window, Nl : int):
        self.x = dict([(i,j) for i, j in enumerate(x)])
        self.Nx = x.shape[0]
        self.Nb = Nb
        self.Nr = Nr
        self.Nt = Nt
        self.Nw = Nw
        self.window = window
        self.Nl = Nl
        
        # test bounds 
        self.smin = 0
        self.smax = (self.Nt + self.Nx) // self.Nr
        
        # Output sequence
        self.y = [np.zeros((self.smax + 1) * self.Nr) for l in range(self.Nl)]
       
        # Assert that things make sense
        assert sp.signal.check_COLA(self.window, self.Nt, self.Nt - self.Nr), f"Input window not COLA for Nt = {Nt} and Nr = {Nr}"
        assert Nr < Nb - Nw + 1, f"Truncation size Nr = {Nr} invalid for filter size of Nw = {Nw}"

    def get_input_block(self, s): 
        assert s >= self.smin, f"s = {s} too small! smin = {self.smin}"
        assert s <= self.smax, f"s = {s} too large! smax = {self.smax}"

        iidx = np.arange(-self.Nb + (s + 1) * self.Nr, (s + 1) * self.Nr)
        return np.array([self.x.get(i, 0) for i in iidx])
        
    def set_output_block(self, s, w):
        assert s >= self.smin, f"s = {s} too small! smin = {self.smin}"
        assert s <= self.smax, f"s = {s} too large! smax = {self.smax}"

        assert len(w) == self.Nl
        oidx = np.arange((s + 1) * self.Nr - self.Nt, (s + 1) * self.Nr)
        u = np.fft.rfft(self.get_input_block(s))
        
        # Compute new frame
        v = [
            np.fft.irfft(u * np.fft.rfft(np.pad(w[l], (0, self.Nb - self.Nw))))[-self.Nt :] for l in range(self.Nl)
        ]
        
        # Add frame to current computation
        for l in range(self.Nl):
            self.y[l][oidx] += (self.window * v[l])
        
        return v