from pcsz.fft.fftmat import minimal_real_vector, iminimal_real_vector, minimal_real_matrix, srfftmat, isrfftmat

import numpy as np
import scipy as sp
import scipy.sparse

class physical_pressure:
    def test(self):
        None
        
    def __init__(self, b_rir, d_rir, Nb, Nv, Nw):
        # Map input quantities
        self.Nb = Nb
        self.Nv = Nv
        self.Na = Nb - Nv
        self.Nw = Nw

        self.b_rir = b_rir
        self.d_rir = d_rir

        self.Nh = self.b_rir.shape[2]
        self.Nl = self.b_rir.shape[1]
        self.Np = self.b_rir.shape[0]

        # Precompute
        self.b_rir_transform = self._rir_transform_matrix(self.b_rir)
        self.d_rir_transform = self._rir_transform_matrix(self.d_rir)

    def b_pressure_vector(self, u, w):
        return self._pressure_vector(u, w, self.b_rir)
    
    def d_pressure_vector(self, u, w):
        return self._pressure_vector(u, w, self.d_rir)

    def _pressure_vector(self, u, w, rir): 
        uc = np.fft.rfft(u)

        wc = np.array([
                np.fft.rfft(
                    np.pad(w[l], (0, self.Nb - self.Nw))
                )
                for l in range(self.Nl)
            ])

        rirc = np.array([
                [
                    np.fft.rfft(
                        np.pad(rir[m][l], (0, self.Nb - self.Nh))
                    )
                    for l in range(self.Nl)
                ]
                for m in range(self.Np)
            ])

        return np.array([
            minimal_real_vector(
                np.fft.rfft(
                    np.fft.irfft(
                        uc * sum([
                            rirc[m][l] * wc[l] for l in range(self.Nl)
                        ])          
                    )[self.Na : ]
                )
            , self.Nv)
            for m in range(self.Np)
        ])

    def b_pressure_matrix(self, u):
        return self._pressure_matrix(u, self.b_rir_transform)
    
    def d_pressure_matrix(self, u):
        return self._pressure_matrix(u, self.d_rir_transform)

    def forward_matrix(self):
        return sp.sparse.block_diag([
            srfftmat(self.Nb) @ \
            np.pad(
                np.eye(self.Nw), 
                ((0, self.Nb - self.Nw), (0, 0))
            ) @ \
            isrfftmat(self.Nw)
            for l in range(self.Nl)
        ])

    def _pressure_matrix(self, u, rir_transform): 
        uc = minimal_real_matrix(
                sp.sparse.diags(np.fft.rfft(u)).tocsr()
            )

        mat = np.array([
                minimal_real_vector(
                    np.fft.rfft(
                        np.fft.irfft(
                            iminimal_real_vector((uc @ rir_transform[m]).toarray(), self.Nb)
                            , n=self.Nb, axis=0)[self.Na : ]
                    , axis=0)
                , self.Nv)
            for m in range(self.Np)
        ])

        return mat

    def _rir_transform_matrix(self, rir):
        forward_matrix = self.forward_matrix()
        rir_transform = [
            sp.sparse.hstack([
                minimal_real_matrix(
                    sp.sparse.diags(
                        np.fft.rfft(
                            np.pad(rir[m][l], (0, self.Nb - self.Nh))
                        )
                    ).tocsr()
                ) 
                for l in range(self.Nl)
            ]) @ forward_matrix 
            for m in range(self.Np)
        ]

        return rir_transform

