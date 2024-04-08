from pcsz.fft.fftmat import minimal_real_vector, iminimal_real_vector, minimal_real_matrix, srfftmat, isrfftmat
from pcsz.pressure.physical_pressure import physical_pressure

import numpy as np
import scipy as sp
import scipy.sparse
import pydetectability as pd

class par_pressure(physical_pressure):
    def __init__(self, b_rir, d_rir, Nb, Nv, Nw, fs, sig_fs, spl_fs, training_rate=1000, N_filters=64):
        # Inherit from parent 
        super().__init__(b_rir, d_rir, Nb, Nv, Nw)

        # Perceptual model
        mapping = pd.signal_pressure_mapping(sig_fs, spl_fs)
        print(f"Using training rate = {training_rate}")
        self.model = pd.par_model(fs, Nv, mapping, training_rate=training_rate, N_filters=N_filters)

    def b_perceptual_pressure_vector(self, u_b, w): 
        return self._perceptual_pressure_vector(u_b, w, self.b_rir, u_b)

    def d_perceptual_pressure_vector(self, u_b, w, u_d): 
        return self._perceptual_pressure_vector(u_b, w, self.d_rir, u_d)

    def b_gain(self, u, w):
        p_x = self._pressure_vector(u, w, self.b_rir)
        g_x = np.array(
            [
                self.model.gain(
                    np.fft.irfft(iminimal_real_vector(p_x[m], self.Nb - self.Na))
                ) for m in range(self.Np)
            ]
        )

        return g_x

    def d_gain(self, u, w):
        p_x = self._pressure_vector(u, w, self.d_rir)
        g_x = np.array(
            [
                self.model.gain(
                    np.fft.irfft(iminimal_real_vector(p_x[m], self.Nb - self.Na))
                ) for m in range(self.Np)
            ]
        )

        return g_x

    def _perceptual_pressure_vector(self, u, w, rir, x): 
        p_x = self._pressure_vector(x, w, rir)
        g_x = np.array(
            [
                self.model.gain(
                    np.fft.irfft(iminimal_real_vector(p_x[m], self.Nb - self.Na))
                ) for m in range(self.Np)
            ]
        )
        # print(f"min :: {np.min(g_x)}")
        # print(f"max :: {np.max(g_x)}")
        # print(f"val :: {g_x[0:5]}")

        G_x = [
                minimal_real_matrix(sp.sparse.diags(
                    g_x[m], format="csr"
                ))
                for m in range(self.Np)
            ]
        p_u = self._pressure_vector(u, w, rir)
        return np.array([
                G_x[m] @ p_u[m]
                for m in range(self.Np)
            ])
        
    def b_perceptual_pressure_matrix(self, u_b, w):
        return self._perceptual_pressure_matrix(u_b, self.b_rir, self.b_rir_transform, w, u_b)
    
    def d_perceptual_pressure_matrix(self, u_b, w, u_d):
        return self._perceptual_pressure_matrix(u_b, self.d_rir, self.d_rir_transform, w, u_d)

    def _perceptual_pressure_matrix(self, u, rir, rir_transform, w, x): 
        p_x = self._pressure_vector(x, w, rir)
        g_x = np.array(
                [
                    self.model.gain(
                        np.fft.irfft(iminimal_real_vector(p_x[m], self.Nb - self.Na))
                    ) for m in range(self.Np)
                ]
            )
        # print(f"min :: {np.min(g_x)}")
        # print(f"max :: {np.max(g_x)}")
        # print(f"val :: {g_x[0:5]}")

        G_x = [
                minimal_real_matrix(sp.sparse.diags(
                    g_x[m], format="csr"
                ))
                for m in range(self.Np)
            ]
        P_u = self._pressure_matrix(u, rir_transform)
        return np.array([
                G_x[m] @ P_u[m]
                for m in range(self.Np)
            ])
