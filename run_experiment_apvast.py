#!/bin/python3

from pcsz.rir.generate_rir import generate_rir
from pcsz.utils.get_logger import get_logger
from pcsz.utils.load_wav import load_wav
from pcsz.defaults.room import room
from pcsz.signal.SIMO_overlap_save import SIMO_overlap_save
from pcsz.signal.trapezoidal_window import trapezoidal_window
from pcsz.pressure.apvast_implementation import apvast
from pcsz.optimize.generate_soundzone_mosek_task import generate_soundzone_mosek_task
from pcsz.optimize.run_soundzone_mosek_task import run_soundzone_mosek_task
from pcsz.fft.fftmat import iminimal_real_vector, minimal_real_vector

import mosek as mk 
import numpy as np 
import scipy as sp
import scipy.signal
import scipy.io.wavfile
import scipy.io
import matplotlib.pyplot as plt
import pathlib
import argparse

if __name__ == "__main__":
    # parse input arguments
    parser = argparse.ArgumentParser("research-pcsz-program")
    parser.add_argument("--fs", type=int, required=True)
    parser.add_argument("--b_filename", type=str, required=True)
    parser.add_argument("--d_filename", type=str, required=True)
    parser.add_argument("--tx", type=float, required=True)
    parser.add_argument("--Nr", type=int, required=True)
    parser.add_argument("--eigenvalue_interval", type=int, required=True)
    args = parser.parse_args()

    logger = get_logger(f"pcsz_apvast_{args.b_filename}_{args.d_filename}")

    # Sampling rate 
    fs = args.fs
    logger.info(f"Running with fs = {fs} hz")

    
    # RIR length
    th = 0.2
    logger.info(f"Running with th = {th} seconds")
    Nh = int(th * fs)
    logger.info(f"Running with Nh = {Nh} samples")
    
    # Filter length
    tw = 0.05
    logger.info(f"Running with tw = {tw} seconds")
    Nw = int(tw * fs)
    logger.info(f"Running with Nw = {Nw} samples")

    # Modeling delay
    modeling_delay = 50
    logger.info(f"Running with modeling_delay = {modeling_delay} samples")
 
    # Valid size length
    Nr = args.Nr
    logger.info(f"Running with Nr = {Nr} samples")

    # Block size
    Nb = 2 * Nr
    logger.info(f"Running with Nb = {Nb} samples")

    # Target song length
    tx = args.tx
    logger.info(f"Running with tx = {tx} seconds")
    Nx = int(tx * fs)
    Nx = Nx - (Nx % Nr)
    logger.info(f"Running with Nx = {Nx} samples")

    # Make room
    room = room() 
    Ns = room.speaker_coords.shape[0]
    Np = room.b_control_coords.shape[0]
    Nc = room.b_validation_coords.shape[0]
    ro = 5
    rt = 0.2
    b_control_rir = generate_rir(Nh, fs, room.b_control_coords, room.speaker_coords, room.room_dim, rt, ro)
    d_control_rir = generate_rir(Nh, fs, room.d_control_coords, room.speaker_coords, room.room_dim, rt, ro)
    b_validation_rir = generate_rir(Nh, fs, room.b_validation_coords, room.speaker_coords, room.room_dim, rt, ro)
    d_validation_rir = generate_rir(Nh, fs, room.d_validation_coords, room.speaker_coords, room.room_dim, rt, ro)
    logger.info(f"Running with Ns = {Ns} speakers")
    logger.info(f"Running with Np = {Np} control points")
    logger.info(f"Running with Nc = {Nc} validation points")
    logger.info(f"Running with ro = {ro}")
    logger.info(f"Running with rt = {rt} seconds")

    # Eigenvalue interval
    eigenvalue_count = Nw * Ns
    eigenvalue_interval = args.eigenvalue_interval 
    logger.info(f"Running with eigenvalue_count = {eigenvalue_count}")
    logger.info(f"Running with eigenvalue_interval = {eigenvalue_interval}")

    # Filenames
    b_filename = args.b_filename
    d_filename = args.d_filename
    logger.info(f"Zone 1 filename = {b_filename}")
    logger.info(f"Zone 2 filename = {d_filename}")

    logger.info(f"Loading wavs...")
    x_b = load_wav(b_filename, fs, Nx)
    x_d = load_wav(d_filename, fs, Nx)

    # NOTE temporary: use white noise instead
    # x_b = np.random.randn(x_b.shape[0])
    # x_d = np.random.randn(x_d.shape[0])

    logger.info(f"Making apvast object...")
    rir_b = np.transpose(b_control_rir, (2,1,0))
    rir_d = np.transpose(d_control_rir, (2,1,0))

    vast = apvast(
        block_size=Nb,
        rir_A=rir_b,
        rir_B=rir_d,
        filter_length=Nw,
        modeling_delay=modeling_delay,
        reference_index_A=0,
        reference_index_B=0,
        number_of_eigenvectors=eigenvalue_count,
        statistics_buffer_length=Nb,
        mu=1.0,
        hop_size=Nr,
        sampling_rate=fs,
        run_A=True,
        run_B=False,
        perceptual=True,
    )

    logger.info("Injecting noise initial noise into the pipeline")
    vast.process_input_buffers(1e-3 * np.random.randn(Nr), 1e-3 * np.random.randn(Nr))
    vast.process_input_buffers(1e-3 * np.random.randn(Nr), 1e-3 * np.random.randn(Nr))

    # Output buffers
    y_b = np.zeros((eigenvalue_count // eigenvalue_interval, Ns, x_b.shape[0]))
    y_b_t = np.zeros((eigenvalue_count // eigenvalue_interval, Ns, x_b.shape[0]))
    y_d_t = np.zeros((eigenvalue_count // eigenvalue_interval, Ns, x_b.shape[0]))

    print(Nr)
    assert Nx % Nr == 0
    for k in range(Nx // Nr):
        logger.info(f"{k + 1} / {Nx // Nr}")
        i_b = x_b[Nr * k : Nr * (k + 1)]
        i_d = x_d[Nr * k : Nr * (k + 1)]
        o_b, o_d, o_b_t, o_d_t = vast.process_input_buffers(i_b, i_d)
        for i in range(0, eigenvalue_count, eigenvalue_interval):
            y_b[i // eigenvalue_interval, :, Nr * k : Nr * (k + 1)] = o_b[i].T
            y_b_t[i // eigenvalue_interval, :, Nr * k : Nr * (k + 1)] = o_b_t[i].T
            y_d_t[i // eigenvalue_interval, :, Nr * k : Nr * (k + 1)] = o_d_t[i].T

    for i in range(0, eigenvalue_count, eigenvalue_interval):
        name = f"apvast_{0.08}_{float(i)}_{pathlib.Path(b_filename).stem}_{pathlib.Path(d_filename).stem}"
        output_path = pathlib.Path(f"./output/{name}")
        output_path.mkdir(parents=True, exist_ok=True)

        gain = np.amax(np.maximum(x_b, x_d)) / 7 
        def wavwrite(filename, x):
            logger.info(f"Writing {filename}...")
            sp.io.wavfile.write(filename, fs, x / gain)

        bright_control_path = (output_path / "bright_control")
        bright_control_path.mkdir(parents=True, exist_ok=True)
        bright_control_path_target = (bright_control_path / "target")
        bright_control_path_target.mkdir(parents=True, exist_ok=True)
        bright_control_path_output = (bright_control_path / "output")
        bright_control_path_output.mkdir(parents=True, exist_ok=True)
        bright_control_path_total = (bright_control_path / "total")
        bright_control_path_total.mkdir(parents=True, exist_ok=True)

        bright_validation_path = (output_path / "bright_validation")
        bright_validation_path.mkdir(parents=True, exist_ok=True)
        bright_validation_path_target = (bright_validation_path / "target")
        bright_validation_path_target.mkdir(parents=True, exist_ok=True)
        bright_validation_path_output = (bright_validation_path / "output")
        bright_validation_path_output.mkdir(parents=True, exist_ok=True)
        bright_validation_path_total = (bright_validation_path / "total")
        bright_validation_path_total.mkdir(parents=True, exist_ok=True)

        dark_control_path = (output_path / "dark_control")
        dark_control_path.mkdir(parents=True, exist_ok=True)
        dark_control_path_target = (dark_control_path / "target")
        dark_control_path_target.mkdir(parents=True, exist_ok=True)
        dark_control_path_output = (dark_control_path / "output")
        dark_control_path_output.mkdir(parents=True, exist_ok=True)
        dark_control_path_total = (dark_control_path / "total")
        dark_control_path_total.mkdir(parents=True, exist_ok=True)

        dark_validation_path = (output_path / "dark_validation")
        dark_validation_path.mkdir(parents=True, exist_ok=True)
        dark_validation_path_target = (dark_validation_path / "target")
        dark_validation_path_target.mkdir(parents=True, exist_ok=True)
        dark_validation_path_output = (dark_validation_path / "output")
        dark_validation_path_output.mkdir(parents=True, exist_ok=True)
        dark_validation_path_total = (dark_validation_path / "total")
        dark_validation_path_total.mkdir(parents=True, exist_ok=True)
        
        for m in range(Np):
            wavwrite(bright_control_path_target / f"{m}.wav", 
                sum([
                    np.convolve(y_b_t[i // eigenvalue_interval, l], b_control_rir[m][l])
                    for l in range(Ns)
                ])
            )
            wavwrite(bright_control_path_output/ f"{m}.wav", 
                sum([
                    np.convolve(y_b[i // eigenvalue_interval, l], b_control_rir[m][l])
                    for l in range(Ns)
                ])
            )
            wavwrite(bright_control_path_total/ f"{m}.wav", 
                sum([
                    np.convolve(y_b[i // eigenvalue_interval, l] + y_d_t[i // eigenvalue_interval, l], b_control_rir[m][l])
                    for l in range(Ns)
                ])
            )
            wavwrite(bright_validation_path_target / f"{m}.wav", 
                sum([
                    np.convolve(y_b_t[i // eigenvalue_interval, l], b_validation_rir[m][l])
                    for l in range(Ns)
                ])
            )
            wavwrite(bright_validation_path_output/ f"{m}.wav", 
                sum([
                    np.convolve(y_b[i // eigenvalue_interval, l], b_validation_rir[m][l])
                    for l in range(Ns)
                ])
            )
            wavwrite(bright_validation_path_total/ f"{m}.wav", 
                sum([
                    np.convolve(y_b[i // eigenvalue_interval, l] + y_d_t[i // eigenvalue_interval, l], b_validation_rir[m][l])
                    for l in range(Ns)
                ])
            )

            wavwrite(dark_control_path_target / f"{m}.wav", 
                sum([
                    np.convolve(y_d_t[i // eigenvalue_interval, l], d_control_rir[m][l])
                    for l in range(Ns)
                ])
            )
            wavwrite(dark_control_path_output/ f"{m}.wav", 
                sum([
                    np.convolve(y_b[i // eigenvalue_interval, l], d_control_rir[m][l])
                    for l in range(Ns)
                ])
            )
            wavwrite(dark_control_path_total/ f"{m}.wav", 
                sum([
                    np.convolve(y_b[i // eigenvalue_interval, l] + y_d_t[i // eigenvalue_interval, l], d_control_rir[m][l])
                    for l in range(Ns)
                ])
            )
            wavwrite(dark_validation_path_target / f"{m}.wav", 
                sum([
                    np.convolve(y_d_t[i // eigenvalue_interval, l], d_validation_rir[m][l])
                    for l in range(Ns)
                ])
            )
            wavwrite(dark_validation_path_output/ f"{m}.wav", 
                sum([
                    np.convolve(y_b[i // eigenvalue_interval, l], d_validation_rir[m][l])
                    for l in range(Ns)
                ])
            )
            wavwrite(dark_validation_path_total/ f"{m}.wav", 
                sum([
                    np.convolve(y_b[i // eigenvalue_interval, l] + y_d_t[i // eigenvalue_interval, l], d_validation_rir[m][l])
                    for l in range(Ns)
                ])
            )
