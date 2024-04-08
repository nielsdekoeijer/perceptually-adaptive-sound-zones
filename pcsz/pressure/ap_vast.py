import pydetectability as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
import scipy as sp
import scipy.sparse
import scipy.linalg

from .apvast_implementation import apvast

# An odd-ball, but essentially we implement AP-VAST here
class ap_vast:
    def __init__(self, Nb, rir_b, rir_d, Nw, modeling_delay, reference_index_A, Nr, fs):


        """
        # rir_b = rir_b[:,:,0:150]
        # rir_d = rir_d[:,:,0:150]

        # rir_b = rir_b + 1e-4 * np.random.randn(rir_b.shape[0], rir_b.shape[1], rir_b.shape[2])
        # rir_d = rir_d + 1e-4 * np.random.randn(rir_d.shape[0], rir_d.shape[1], rir_d.shape[2])
        self.it = 0
        for i in range(rir_b.shape[0]):
            for j in range(rir_b.shape[1]):
                plt.plot(rir_b[i][j])
                plt.plot(rir_d[i][j])
                plt.savefig(f"{i},{j}.png")
                plt.close()
        """

        rir_b = np.transpose(rir_b, (2,1,0))
        rir_d = np.transpose(rir_d, (2,1,0))

        self.Nr = Nr
        self.Nw = Nw
        self.Nl = rir_b.shape[1]
        self.Neig = self.Nl * self.Nw
        self.vast = apvast(
               block_size=2 * Nr, 
               rir_A=rir_b, 
               rir_B=rir_d, 
               filter_length=Nw, 
               modeling_delay=50,
               reference_index_A=reference_index_A,
               reference_index_B=reference_index_A,
               mu=1.0,
               statistics_buffer_length=2 * Nr,
               number_of_eigenvectors=Nw * 10,
               hop_size=Nr,
               sampling_rate=fs,
               run_A=True,
               run_B=False,
               perceptual=True,
           )

        print("injecting noise for good starting point 1...")
        self.vast.process_input_buffers(1e-3 * np.random.randn(self.Nr), 1e-3 * np.random.randn(self.Nr))
        print("injecting noise for good starting point 2...")
        self.vast.process_input_buffers(1e-3 * np.random.randn(self.Nr), 1e-3 * np.random.randn(self.Nr))

    # Slightly different implementation due to AP-VAST
    def find_filter(self, u_b, u_d, w_t):
        self.vast.process_input_buffers(u_b[-self.Nr:], u_d[-self.Nr:])
        # self.vast.process_input_buffers(1e-3 * np.random.randn(self.Nr), 1e-3 * np.random.randn(self.Nr))
        wp = self.vast.w_A.squeeze(-1)

        """
        plt.plot(self.vast.loudspeaker_weighted_response_A_to_A_buffer[:,0,0])
        plt.plot(self.vast.loudspeaker_weighted_response_A_to_B_buffer[:,0,0])
        plt.plot(self.vast.loudspeaker_weighted_response_B_to_A_buffer[:,0,0])
        plt.plot(self.vast.loudspeaker_weighted_response_B_to_B_buffer[:,0,0])
        plt.savefig(f"{self.it}")
        plt.close()
        self.it += 1
        """

        w = []
        for i in range(self.Neig):
            wk = []
            for l in range(self.Nl):
                wk.append(wp[i][l * self.Nw: (l + 1) * self.Nw])
            w.append(wk)

        return w
