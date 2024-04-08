from pcsz.fft.fftmat import minimal_real_vector, iminimal_real_vector, minimal_real_matrix, srfftmat, isrfftmat
from pcsz.pressure.physical_pressure import physical_pressure

import numpy as np
import scipy as sp
import scipy.sparse
import scipy.linalg
import pydetectability as pd

class taal_pressure(physical_pressure):
    def __init__(self, b_rir, d_rir, Nb, Nv, Nw, fs, sig_fs, spl_fs, training_rate=1000, N_filters=64):
        # Inherit from parent 
        super().__init__(b_rir, d_rir, Nb, Nv, Nw)

        # Perceptual model
        self.Ntaal = N_filters
        self.Np = b_rir.shape[0]

        mapping = pd.signal_pressure_mapping(sig_fs, spl_fs)
        print(f"Using training rate = {training_rate}")
        self.model = pd.taal_model(fs, Nv, mapping, N_filters=self.Ntaal, training_rate=training_rate)
        self.internal_filter = np.array([
                minimal_real_matrix(np.diag(self.model.auditory_filter_bank_freq[i])).toarray()
                for i in range(self.Ntaal)
            ])

    def b_perceptual_pressure_vector(self, u_b, w): 
        return self._perceptual_pressure_vector(u_b, w, self.b_rir, u_b)

    def d_perceptual_pressure_vector(self, u_b, w, u_d): 
        return self._perceptual_pressure_vector(u_b, w, self.d_rir, u_d)

    def gain_matrix(self, g_x):
        return np.array([
            [
                np.diag(g_x[m][i]) @ isrfftmat(self.Nv)
                @ self.internal_filter[i]
                for i in range(self.Ntaal)
            ]
            for m in range(self.Np)
        ])
    
    def to_sqrtm_form(self, G_xi):
        return np.array([
            sp.linalg.sqrtm(
                sum([
                    np.matmul(G_xi[m][i].T, G_xi[m][i])
                    for i in range(self.Ntaal)
                ])
            )
            for m in range(self.Np)
        ])


    def _perceptual_pressure_vector(self, u, w, rir, x): 
        p_x = self._pressure_vector(x, w, rir)
        g_x = np.array(
            [
                self.model.gain(
                    np.fft.irfft(iminimal_real_vector(p_x[m], self.Nb - self.Na), n=self.Nb - self.Na)
                ) for m in range(self.Np)
            ]
        )

        G_xi = self.gain_matrix(g_x)
        W_mi = self.to_sqrtm_form(G_xi)

        print(np.sum(np.imag(W_mi)))
        W_mi = np.real(W_mi) 

        p_u = self._pressure_vector(u, w, rir)

        return np.array([
                W_mi[m] @ p_u[m]
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

        G_xi = self.gain_matrix(g_x)
        W_mi = self.to_sqrtm_form(G_xi)

        print(np.sum(np.imag(W_mi)))
        W_mi = np.real(W_mi) 

        P_u = self._pressure_matrix(u, rir_transform)
        return np.array([
                W_mi[m] @ P_u[m]
                for m in range(self.Np)
            ])
