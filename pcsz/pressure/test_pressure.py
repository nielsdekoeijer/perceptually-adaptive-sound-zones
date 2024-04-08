from pcsz.pressure.physical_pressure import physical_pressure
from pcsz.pressure.par_pressure import par_pressure
from pcsz.pressure.taal_pressure import taal_pressure
from pcsz.fft.fftmat import minimal_real_vector, iminimal_real_vector, srfftmat

import pytest
import numpy as np
import scipy as sp

Nb = 1024
fs = 8000
Nv = 200
Nh = 400
Nw = 100
Np = 3
Nl = 2

u_b = np.random.randn(Nb)
u_d = np.random.randn(Nb)
w = np.array([np.random.randn(Nw) for l in range(Nl)])
ws = np.hstack([
    minimal_real_vector(np.fft.rfft(w[l]), Nw) for l in range(Nl)
])

b_rir = np.array([
    [
        np.random.randn(Nh)
        for l in range(Nl)
    ]
    for m in range(Np)
])

d_rir = np.array([
    [
        np.random.randn(Nh)
        for l in range(Nl)
    ]
    for m in range(Np)
])

phys_pressure = physical_pressure(b_rir, d_rir, Nb, Nv, Nw)
par_pressure = par_pressure(b_rir, d_rir, Nb, Nv, Nw, fs, 1, 100)
taal_pressure = taal_pressure(b_rir, d_rir, Nb, Nv, Nw, fs, 1, 100)

@pytest.mark.parametrize("pressure", [phys_pressure, par_pressure, taal_pressure])
def test_forward_matrix(pressure):
    pressure.test()
    forward_matrix = pressure.forward_matrix()
    assert np.linalg.norm(
        forward_matrix @ ws - \
        np.hstack([
            minimal_real_vector(np.fft.rfft(np.pad(w[l], (0, Nb - Nw))), Nb)
            for l in range(Nl)
        ])
    ) < 1e-8

@pytest.mark.parametrize("pressure", [phys_pressure, par_pressure, taal_pressure])
def test_rir_transform(pressure):
    wc = np.array([
            np.fft.rfft(
                np.pad(w[l], (0, Nb - Nw))
            )
            for l in range(Nl)
        ])

    b_rirc = np.array([
            [
                np.fft.rfft(
                    np.pad(b_rir[m][l], (0, Nb - Nh))
                )
                for l in range(Nl)
            ]
            for m in range(Np)
        ])

    d_rirc = np.array([
            [
                np.fft.rfft(
                   np.pad(d_rir[m][l], (0, Nb - Nh))
                )
                for l in range(Nl)
            ]
            for m in range(Np)
        ])

    b_rir_transform = pressure.b_rir_transform
    for m in range(Np):
        assert np.linalg.norm(
            b_rir_transform[m] @ ws - \
            minimal_real_vector(sum([
                b_rirc[m][l] * wc[l] for l in range(Nl)
            ]), Nb)
        ) < 1e-8

    d_rir_transform = pressure.d_rir_transform
    for m in range(Np):
        assert np.linalg.norm(
            d_rir_transform[m] @ ws - \
            minimal_real_vector(sum([
                d_rirc[m][l] * wc[l] for l in range(Nl)
            ]), Nb)
        ) < 1e-8

@pytest.mark.parametrize("pressure", [phys_pressure, par_pressure, taal_pressure])
def test_pressure(pressure):
    b_pressure_matrix = pressure.b_pressure_matrix(u_b)
    b_pressure_vector = pressure.b_pressure_vector(u_b, w)
    for m in range(Np):
        assert np.linalg.norm(b_pressure_vector[m] - b_pressure_matrix[m] @ ws) < 1e-8

    d_pressure_matrix = pressure.d_pressure_matrix(u_b)
    d_pressure_vector = pressure.d_pressure_vector(u_b, w)
    for m in range(Np):
        assert np.linalg.norm(d_pressure_vector[m] - d_pressure_matrix[m] @ ws) < 1e-8

@pytest.mark.parametrize("pressure", [par_pressure, taal_pressure])
def test_perceptual_pressure(pressure):
    b_perceptual_pressure_matrix = pressure.b_perceptual_pressure_matrix(u_b, w)
    b_perceptual_pressure_vector = pressure.b_perceptual_pressure_vector(u_b, w)
    for m in range(Np):
        assert np.linalg.norm(b_perceptual_pressure_vector[m] - b_perceptual_pressure_matrix[m] @ ws) < 1e-8

    d_perceptual_pressure_matrix = pressure.d_perceptual_pressure_matrix(u_b, w, u_d)
    d_perceptual_pressure_vector = pressure.d_perceptual_pressure_vector(u_b, w, u_d)
    for m in range(Np):
        assert np.linalg.norm(d_perceptual_pressure_vector[m] - d_perceptual_pressure_matrix[m] @ ws) < 1e-8

@pytest.mark.parametrize("pressure", [phys_pressure, par_pressure, taal_pressure])
def test_pressure_vector(pressure):
    b_pressure_vector = pressure.b_pressure_vector(u_b, w)
    b_pressure_vector_time = np.array([
            np.fft.irfft(iminimal_real_vector(b_pressure_vector[m], Nv)) 
            for m in range(Np)    
        ]) 
    b_pressure_vector_time_ref = np.array([
            np.fft.irfft(
                sum([
                    np.fft.rfft(
                        np.pad(b_rir[m][l], (0, Nb - Nh))
                    ) * \
                    np.fft.rfft(
                        np.pad(w[l], (0, Nb - Nw))
                    ) * \
                    np.fft.rfft(
                        u_b
                    )
                    for l in range(Nl)
                ]))[-Nv :]
            for m in range(Np)
        ])

    assert np.linalg.norm(
        b_pressure_vector_time - b_pressure_vector_time_ref
    ) / np.linalg.norm(b_pressure_vector_time_ref) < 1e-8

    d_pressure_vector = pressure.d_pressure_vector(u_d, w)
    d_pressure_vector_time = np.array([
            np.fft.irfft(iminimal_real_vector(d_pressure_vector[m], Nv)) 
            for m in range(Np)    
        ]) 
    d_pressure_vector_time_ref = np.array([
            np.fft.irfft(
                sum([
                    np.fft.rfft(
                        np.pad(d_rir[m][l], (0, Nb - Nh))
                    ) * \
                    np.fft.rfft(
                        np.pad(w[l], (0, Nb - Nw))
                    ) * \
                    np.fft.rfft(
                        u_d
                    )
                    for l in range(Nl)
                ]))[-Nv :]
            for m in range(Np)
        ])

    assert np.linalg.norm(
        d_pressure_vector_time - d_pressure_vector_time_ref
    ) / np.linalg.norm(d_pressure_vector_time_ref) < 1e-8
    
@pytest.mark.parametrize("pressure", [par_pressure])
def test_par_pressure_vector(pressure):
    b_pressure_vector = pressure.b_pressure_vector(u_b, w)
    b_pressure_vector_time = np.array([
            np.fft.irfft(iminimal_real_vector(b_pressure_vector[m], Nv)) 
            for m in range(Np)    
        ]) 
    g_b = np.array([pressure.model.gain(b_pressure_vector_time[m]) for m in range(Np)])

    b_perceptual_pressure_vector = pressure.b_perceptual_pressure_vector(u_b, w)
    b_perceptual_pressure_vector_ref = np.array([
            minimal_real_vector(g_b[m] * iminimal_real_vector(b_pressure_vector[m], Nv), Nv)
            for m in range(Np)    
        ]) 

    assert np.linalg.norm(b_perceptual_pressure_vector - b_perceptual_pressure_vector_ref ) / \
        np.linalg.norm(b_perceptual_pressure_vector_ref) < 1e-8

    d_pressure_vector_d = pressure.d_pressure_vector(u_d, w)
    d_pressure_vector_b = pressure.d_pressure_vector(u_b, w)
    d_pressure_vector_time = np.array([
            np.fft.irfft(iminimal_real_vector(d_pressure_vector_d[m], Nv)) 
            for m in range(Np)    
        ]) 
    g_d = np.array([pressure.model.gain(d_pressure_vector_time[m]) for m in range(Np)])

    d_perceptual_pressure_vector = pressure.d_perceptual_pressure_vector(u_b, w, u_d)
    d_perceptual_pressure_vector_ref = np.array([
            minimal_real_vector(g_d[m] * iminimal_real_vector(d_pressure_vector_b[m], Nv), Nv)
            for m in range(Np)    
        ]) 

    assert np.linalg.norm(d_perceptual_pressure_vector - d_perceptual_pressure_vector_ref ) / \
        np.linalg.norm(d_perceptual_pressure_vector_ref) < 1e-8
    
@pytest.mark.parametrize("pressure", [par_pressure])
def test_par_pressure_detectability(pressure):
    b_pressure_vector = pressure.b_pressure_vector(u_b, w)
    b_pressure_vector_time = np.array([
            np.fft.irfft(iminimal_real_vector(b_pressure_vector[m], Nv)) 
            for m in range(Np)    
        ]) 
    d_b_ref = np.array([pressure.model.detectability_gain(b_pressure_vector_time[m], b_pressure_vector_time[m]) for m in range(Np)])

    b_perceptual_pressure_vector = pressure.b_perceptual_pressure_vector(u_b, w)
    d_b = np.array([np.power(np.linalg.norm(b_perceptual_pressure_vector[m]), 2.0) for m in range(Np)])
    assert np.linalg.norm(d_b - d_b_ref) / \
        np.linalg.norm(d_b_ref) < 1e-8

    d_pressure_vector = pressure.d_pressure_vector(u_d, w)
    d_pressure_vector_time = np.array([
            np.fft.irfft(iminimal_real_vector(d_pressure_vector[m], Nv)) 
            for m in range(Np)    
        ]) 
    d_d_ref = np.array([pressure.model.detectability_gain(d_pressure_vector_time[m], d_pressure_vector_time[m]) for m in range(Np)])

    d_perceptual_pressure_vector = pressure.d_perceptual_pressure_vector(u_d, w, u_d)
    d_d = np.array([np.power(np.linalg.norm(d_perceptual_pressure_vector[m]), 2.0) for m in range(Np)])
    assert np.linalg.norm(d_d - d_d_ref) / \
        np.linalg.norm(d_d_ref) < 1e-8


@pytest.mark.parametrize("pressure", [taal_pressure])
def test_taal_pressure_vector(pressure):
    None
    

@pytest.mark.parametrize("pressure", [taal_pressure])
def test_taal_internal_representation(pressure):
    b_pressure_vector = pressure.b_pressure_vector(u_b, w)
    b_internal_vector_ref = np.array([
        pressure.model._taal_model__apply_auditory_filter_bank(np.fft.irfft(iminimal_real_vector(b_pressure_vector[m], Nv))) for m in range(Np)
    ])
    b_internal_vector = np.array([
        [
            np.fft.irfft(iminimal_real_vector(pressure.internal_filter[i] @ b_pressure_vector[m], Nv), n=Nv)
            for i in range(64)
        ]
        for m in range(Np)
    ])

    assert np.linalg.norm(b_internal_vector - b_internal_vector_ref) / \
        np.linalg.norm(b_internal_vector_ref) < 1e-8

    d_pressure_vector = pressure.d_pressure_vector(u_d, w)
    d_internal_vector_ref = np.array([
        pressure.model._taal_model__apply_auditory_filter_bank(np.fft.irfft(iminimal_real_vector(d_pressure_vector[m], Nv))) for m in range(Np)
    ])
    d_internal_vector = np.array([
        [
            np.fft.irfft(iminimal_real_vector(pressure.internal_filter[i] @ d_pressure_vector[m], Nv), n=Nv)
            for i in range(64)
        ]
        for m in range(Np)
    ])

    assert np.linalg.norm(b_internal_vector - b_internal_vector_ref) / \
        np.linalg.norm(b_internal_vector_ref) < 1e-8

@pytest.mark.parametrize("pressure", [taal_pressure])
def test_taal_gain_matrix_detectability(pressure):
    b_pressure_vector = pressure.b_pressure_vector(u_b, w)
    b_pressure_vector_time = np.array([
            np.fft.irfft(iminimal_real_vector(b_pressure_vector[m], Nv), n=Nv) 
            for m in range(Np)    
        ]) 

    g_b = np.array([pressure.model.gain(b_pressure_vector_time[m]) for m in range(Np)])
    G_b = pressure.gain_matrix(g_b)
    d_b_ref = np.array([pressure.model.detectability_gain(b_pressure_vector_time[m], b_pressure_vector_time[m]) for m in range(Np)])
    d_b = np.array([np.array([np.power(np.linalg.norm(G_b[m][i] @ b_pressure_vector[m]), 2.0) for i in range(64)]).sum() for m in range(Np)])

    assert np.linalg.norm(d_b - d_b_ref) / \
        np.linalg.norm(d_b_ref) < 1e-8

    d_pressure_vector = pressure.d_pressure_vector(u_d, w)
    d_pressure_vector_time = np.array([
            np.fft.irfft(iminimal_real_vector(d_pressure_vector[m], Nv), n=Nv) 
            for m in range(Np)    
        ]) 

    g_d = np.array([pressure.model.gain(d_pressure_vector_time[m]) for m in range(Np)])
    G_d = pressure.gain_matrix(g_d)
    d_d_ref = np.array([pressure.model.detectability_gain(d_pressure_vector_time[m], d_pressure_vector_time[m]) for m in range(Np)])
    d_d = np.array([np.array([np.power(np.linalg.norm(G_d[m][i] @ d_pressure_vector[m]), 2.0) for i in range(64)]).sum() for m in range(Np)])

    assert np.linalg.norm(d_d - d_d_ref) / \
        np.linalg.norm(d_d_ref) < 1e-8
    
@pytest.mark.parametrize("pressure", [taal_pressure])
def test_taal_pressure_detectability(pressure):
    b_pressure_vector = pressure.b_pressure_vector(u_b, w)
    b_pressure_vector_time = np.array([
            np.fft.irfft(iminimal_real_vector(b_pressure_vector[m], Nv)) 
            for m in range(Np)    
        ]) 
    d_b_ref = np.array([pressure.model.detectability_gain(b_pressure_vector_time[m], b_pressure_vector_time[m]) for m in range(Np)])

    b_perceptual_pressure_vector = pressure.b_perceptual_pressure_vector(u_b, w)
    d_b = np.array([np.power(np.linalg.norm((b_perceptual_pressure_vector[m])), 2.0) for m in range(Np)])
    assert np.linalg.norm(d_b - d_b_ref) / \
        np.linalg.norm(d_b_ref) < 1e-8

    d_pressure_vector = pressure.d_pressure_vector(u_d, w)
    d_pressure_vector_time = np.array([
            np.fft.irfft(iminimal_real_vector(d_pressure_vector[m], Nv)) 
            for m in range(Np)    
        ]) 
    d_d_ref = np.array([pressure.model.detectability_gain(d_pressure_vector_time[m], d_pressure_vector_time[m]) for m in range(Np)])

    d_perceptual_pressure_vector = pressure.d_perceptual_pressure_vector(u_d, w, u_d)
    d_d = np.array([np.power(np.linalg.norm((d_perceptual_pressure_vector[m])), 2.0) for m in range(Np)])
    assert np.linalg.norm(d_d - d_d_ref) / \
        np.linalg.norm(d_d_ref) < 1e-8


if __name__=="__main__":
    test_taal_internal_representation(taal_pressure)
    test_forward_matrix(phys_pressure)
    test_forward_matrix(par_pressure)
    test_rir_transform(phys_pressure)
    test_rir_transform(par_pressure)
    test_pressure(phys_pressure)
    test_pressure(par_pressure)
    test_perceptual_pressure(par_pressure)
    test_perceptual_pressure(taal_pressure)
    test_pressure_vector(phys_pressure)
    test_pressure_vector(par_pressure)
    test_pressure_vector(taal_pressure)
    test_par_pressure_vector(par_pressure)
    test_par_pressure_detectability(par_pressure)
    test_taal_gain_matrix_detectability(taal_pressure)
    test_taal_pressure_detectability(taal_pressure)
