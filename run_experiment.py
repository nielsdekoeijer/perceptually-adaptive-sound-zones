#!/bin/python3
from pcsz.rir.generate_rir import generate_rir
from pcsz.utils.get_logger import get_logger
from pcsz.utils.load_wav import load_wav
from pcsz.defaults.room import room
from pcsz.signal.SIMO_overlap_save import SIMO_overlap_save
from pcsz.signal.trapezoidal_window import trapezoidal_window
from pcsz.pressure.physical_pressure import physical_pressure
from pcsz.pressure.par_pressure import par_pressure
from pcsz.pressure.ap_vast import ap_vast
from pcsz.pressure.taal_pressure import taal_pressure
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
    # Create logger
    parser = argparse.ArgumentParser("research-pcsz-program")
    parser.add_argument("--fs", type=int, required=True)
    parser.add_argument("--b_filename", type=str, required=True)
    parser.add_argument("--d_filename", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--D0", type=float, required=True)
    parser.add_argument("--tx", type=float, required=True)
    parser.add_argument("--tv", type=float, required=True)
    parser.add_argument("--static_filter", type=str)
    args = parser.parse_args()

    logger = get_logger(f"pcsz_{args.b_filename}_{args.d_filename}_{args.mode}")

    # Sampling rate 
    fs = args.fs
    logger.info(f"Running with fs = {fs} hz")
    
    # Target song length
    tx = args.tx
    logger.info(f"Running with tx = {tx} seconds")
    Nx = int(tx * fs)
    logger.info(f"Running with Nx = {Nx} samples")
    
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
    
    # Valid size length
    tv = args.tv 
    logger.info(f"Running with tv = {tv} seconds")
    Nv = int(tv * fs)
    logger.info(f"Running with Nv = {Nv} samples")
    
    mode = args.mode

    # Block size
    Nb = Nh + Nw + Nv - 2
    logger.info(f"Running with Nb = {Nb} samples")
    
    # Hop size
    Nr = Nv - Nv // 8
    logger.info(f"Running with Nr = {Nr} samples")
    
    # Get room parameters and draw...
    visualize = False
    room = room() 
    if visualize:
        room.draw()

    # Speaker positions
    Ns = room.speaker_coords.shape[0]
    logger.info(f"Running with Ns = {Ns} speakers")
    
    Np = room.b_control_coords.shape[0]
    logger.info(f"Running with Np = {Np} control points")

    Nc = room.b_validation_coords.shape[0]
    logger.info(f"Running with Nc = {Nc} validation points")
    
    ro = 5
    logger.info(f"Running with ro = {ro}")

    rt = 0.2
    logger.info(f"Running with rt = {rt} seconds")

    E0_db = 10.
    E0 = np.power(10, (E0_db/ 10))
    logger.info(f"Running with E0 = {E0_db} dB")
    logger.info(f"Running with E0 = {E0} sg")

    D0 = args.D0
    logger.info(f"Running with D0 = {D0}")

    b_filename = args.b_filename
    logger.info(f"Zone 1 filename = {b_filename}")

    d_filename = args.d_filename
    logger.info(f"Zone 2 filename = {d_filename}")
    
    logger.info(f"Generating window...")
    window = trapezoidal_window(Nv, Nr)

    logger.info(f"Generating RIRs...")
    # Note: in microphone loudspeaker sample
    b_control_rir = generate_rir(Nh, fs, room.b_control_coords, room.speaker_coords, room.room_dim, rt, ro)
    d_control_rir = generate_rir(Nh, fs, room.d_control_coords, room.speaker_coords, room.room_dim, rt, ro)
    b_validation_rir = generate_rir(Nh, fs, room.b_validation_coords, room.speaker_coords, room.room_dim, rt, ro)
    d_validation_rir = generate_rir(Nh, fs, room.d_validation_coords, room.speaker_coords, room.room_dim, rt, ro)

    sp.io.savemat("rirs.mat", {
        "gB" : np.swapaxes(b_control_rir, 2, 1), 
        "gD" : np.swapaxes(d_control_rir, 2, 1),
        "I"  : Nw,
        "fs" : fs
    })

    logger.info(f"Loading wavs...")
    x_b = load_wav(b_filename, fs, Nx)
    x_d = load_wav(d_filename, fs, Nx)

    logger.info(f"Selected mode = {mode}")
    if mode == "apvast":
        Neig = Ns * Nw
        Neig_hop = 100
        logger.info(f"Number of eigenvalues: {Neig}")

    if mode == "apvast":
        s_b = [SIMO_overlap_save(x_b, Nb, Nr, Nv, Nw, window, Ns) for i in range(0, Neig, Neig_hop)]
        s_b_t = [SIMO_overlap_save(x_b, Nb, Nr, Nv, Nw, window, Ns) for i in range(0, Neig, Neig_hop)]
        s_d = [SIMO_overlap_save(x_d, Nb, Nr, Nv, Nw, window, Ns) for i in range(0, Neig, Neig_hop)]
        s_d_t = [SIMO_overlap_save(x_d, Nb, Nr, Nv, Nw, window, Ns) for i in range(0, Neig, Neig_hop)]
    else:
        s_b = SIMO_overlap_save(x_b, Nb, Nr, Nv, Nw, window, Ns)
        s_b_t = SIMO_overlap_save(x_b, Nb, Nr, Nv, Nw, window, Ns)
        s_d = SIMO_overlap_save(x_d, Nb, Nr, Nv, Nw, window, Ns)
        s_d_t = SIMO_overlap_save(x_d, Nb, Nr, Nv, Nw, window, Ns)


    logger.info(f"Creating pressure object...")
    delay = 50
    assert mode == "par" or mode == "taal" or mode == "ref" or mode == "static" or mode == "apvast", f"Mode {mode} unsupported"
    if mode == "ref":
        pressure = physical_pressure(b_control_rir, d_control_rir, Nb, Nv, Nw)
    if mode == "static":
        w_static = sp.io.loadmat(args.static_filter)["staticFilters"].T
    if mode == "par":
        pressure = par_pressure(b_control_rir, d_control_rir, Nb, Nv, Nw, fs, 1, 85)
    if mode == "taal":
        pressure = taal_pressure(b_control_rir, d_control_rir, Nb, Nv, Nw, fs, 1, 85)
    if mode == "apvast":
        pressure = ap_vast(Nb, b_control_rir, d_control_rir, Nw, delay, 0, Nr, fs)

    w_t = np.array([
        np.array([1.0 * ((i == delay) and (l == 0)) for i in range(Nw)]) 
        for l in range(room.speaker_coords.shape[0])
    ])
    w_t_o = np.hstack([
        minimal_real_vector(np.fft.rfft(w_t[l]), Nw)
        for l in range(room.speaker_coords.shape[0])
        ])
    
    # Problem name
    if mode == "static":
        name = f"{mode}_{tv}_{pathlib.Path(args.static_filter).stem}_{pathlib.Path(b_filename).stem}_{pathlib.Path(d_filename).stem}"
    else:
        name = f"{mode}_{tv}_{D0}_{pathlib.Path(b_filename).stem}_{pathlib.Path(d_filename).stem}"

    print("D")

    logger.info(f"Running with name = {name}")

    with mk.Env() as env:
        if mode == "apvast":
            smin = s_b[0].smin
            smax = s_b[0].smax
        else:
            smin = s_b.smin
            smax = s_b.smax

        for s in range(smin, smax + 1):
            logger.info(f"Iteration {s} / {smax}")
            logger.info(f"Creating matrices...")
            if mode == "apvast":
                u_b = s_b[0].get_input_block(s)
                u_d = s_d[0].get_input_block(s)
            else:
                u_b = s_b.get_input_block(s)
                u_d = s_d.get_input_block(s)

            if mode == "ref":
                b_pressure_matrix_u_b = pressure.b_pressure_matrix(u_b)
                d_pressure_matrix_u_b = pressure.d_pressure_matrix(u_b)
                d_pressure_matrix_u_d = pressure.d_pressure_matrix(u_d)

                b_target_pressure_u_b = pressure.b_pressure_vector(u_b, w_t)
                d_target_pressure_u_b = pressure.d_pressure_vector(u_b, w_t)

            if mode == "par" or mode == "taal":
                b_pressure_matrix_u_b = pressure.b_perceptual_pressure_matrix(u_b, w_t)
                d_pressure_matrix_u_b = pressure.d_perceptual_pressure_matrix(u_b, w_t, u_d)
                d_pressure_matrix_u_d = pressure.d_perceptual_pressure_matrix(u_d, w_t, u_d)

                b_target_pressure_u_b = pressure.b_perceptual_pressure_vector(u_b, w_t)
                d_target_pressure_u_b = pressure.d_perceptual_pressure_vector(u_b, w_t, u_d)
            
            if mode == "static" or mode == "apvast":
                if mode == "static":
                    logger.info(f"Setting...")
                    w = w_static

                if mode == "apvast":
                    logger.info(f"Setting...")
                    w = pressure.find_filter(u_b, u_d, w_t)
            else:
                with env.Task(0, 0) as task:
                    # Create task
                    if mode == "ref":
                        task = generate_soundzone_mosek_task(
                            task, 
                            d_pressure_matrix_u_b,
                            b_pressure_matrix_u_b,
                            b_target_pressure_u_b, 
                            [D0 * np.linalg.norm(b_target_pressure_u_b[m]) for m in range(Np)],
                            E0,
                            verbose=True
                        )
                    if mode == "par" or mode == "taal":
                        task = generate_soundzone_mosek_task(
                            task, 
                            d_pressure_matrix_u_b,
                            b_pressure_matrix_u_b,
                            b_target_pressure_u_b, 
                            [np.sqrt(D0)] * Np,  
                            E0,
                            verbose=True
                        )

                    # Run task
                    logger.info(f"Solving...")
                    w_o = run_soundzone_mosek_task(
                        task, 
                        1 + b_pressure_matrix_u_b.shape[0] + b_pressure_matrix_u_b.shape[2],
                        b_pressure_matrix_u_b.shape[2],
                        verbose=True
                    )
                    
                    # Convert to time domain
                    Nwh = b_pressure_matrix_u_b.shape[2] // room.speaker_coords.shape[0]
                    w = [
                        np.fft.irfft(iminimal_real_vector(w_o[l * Nwh : (l + 1) * Nwh], Nw))
                        for l in range(room.speaker_coords.shape[0])
                    ]

            # Store result 
            logger.info(f"w energy: {np.linalg.norm(w)}")
            if mode == "apvast":
                for i in range(0, Neig, Neig_hop):
                    v = s_b[i // Neig_hop].set_output_block(s, w[i])  
                    v_t = s_b_t[i // Neig_hop].set_output_block(s, w_t)    
                    s_d_t[i // Neig_hop].set_output_block(s, w_t)    
            else:
                v = s_b.set_output_block(s, w)    
                v_t = s_b_t.set_output_block(s, w_t)    
                s_d_t.set_output_block(s, w_t)    

            if not mode == "static" and visualize and s > 0 and s % 1 == 0:
                fig, axs = plt.subplots(3, 2, constrained_layout=True)
                axs[0][0].set_title("Total bright pressure")
                axs[0][0].set_xlabel("sample [-]")
                axs[0][0].set_ylabel("pressure [-]")
                axs[0][0].plot(sum([
                    np.convolve(b_validation_rir[0][l], s_b_t.y[l])
                    for l in range(room.speaker_coords.shape[0])
                ]), label="Target Pressure")
                axs[0][0].plot(sum([
                    np.convolve(b_validation_rir[0][l], s_b.y[l])
                    for l in range(room.speaker_coords.shape[0])
                ]), label="Output Pressure")
                axs[0][0].legend()

                axs[1][0].set_title("Total dark pressure")
                axs[1][0].set_xlabel("sample [-]")
                axs[1][0].set_ylabel("pressure [-]")
                axs[1][0].plot(sum([
                    np.convolve(d_validation_rir[0][l], s_b_t.y[l])
                    for l in range(room.speaker_coords.shape[0])
                ]), label="Target Pressure")
                axs[1][0].plot(sum([
                    np.convolve(d_validation_rir[0][l], s_b.y[l])
                    for l in range(room.speaker_coords.shape[0])
                ]), label="Output Pressure")
                axs[1][0].legend()

                axs[0][1].set_title("Current frame bright pressure ")
                axs[0][1].set_xlabel("sample [-]")
                axs[0][1].set_ylabel("pressure [-]")
                axs[0][1].plot(sum([
                    np.fft.irfft(iminimal_real_vector(b_pressure_matrix_u_b[0] @ w_t_o, Nv))
                    for l in range(room.speaker_coords.shape[0])
                ]), label="Target Pressure")
                axs[0][1].plot(sum([
                    np.fft.irfft(iminimal_real_vector(b_pressure_matrix_u_b[0] @ w_o, Nv))
                    for l in range(room.speaker_coords.shape[0])
                ]), label="Output Pressure")
                axs[0][1].legend()

                axs[1][1].set_title("Current frame dark pressure")
                axs[1][1].set_xlabel("sample [-]")
                axs[1][1].set_ylabel("pressure [-]")
                axs[1][1].plot(sum([
                    np.fft.irfft(iminimal_real_vector(d_pressure_matrix_u_b[0] @ w_t_o, Nv))
                    for l in range(room.speaker_coords.shape[0])
                ]), label="Target Pressure")
                axs[1][1].plot(sum([
                    np.fft.irfft(iminimal_real_vector(d_pressure_matrix_u_b[0] @ w_o, Nv))
                    for l in range(room.speaker_coords.shape[0])
                ]), label="Output Pressure")
                axs[1][1].legend()

                axs[2][0].set_title("Direct vs OS Convolution")
                axs[2][0].set_xlabel("sample [-]")
                axs[2][0].set_ylabel("pressure [-]")
                axs[2][0].plot(sum([
                    np.convolve(np.convolve(b_validation_rir[0][l], x_b), w_t[l])
                    for l in range(room.speaker_coords.shape[0])
                ]), label="Target Direct")
                axs[2][0].plot(sum([
                    np.convolve(b_validation_rir[0][l], s_b_t.y[l])
                    for l in range(room.speaker_coords.shape[0])
                ]), label="Target Overlap Save")
                axs[2][0].plot(sum([
                    np.convolve(b_validation_rir[0][l], s_b.y[l])
                    for l in range(room.speaker_coords.shape[0])
                ]), label="Output Overlap Save")
                axs[2][0].legend()

                axs[2][1].set_title("Direct vs OS Convolution")
                axs[2][1].set_xlabel("sample [-]")
                axs[2][1].set_ylabel("pressure [-]")
                axs[2][1].plot(sum([
                    np.convolve(np.convolve(d_validation_rir[0][l], x_b), w_t[l])
                    for l in range(room.speaker_coords.shape[0])
                ]), label="Target Direct")
                axs[2][1].plot(sum([
                    np.convolve(d_validation_rir[0][l], s_b_t.y[l])
                    for l in range(room.speaker_coords.shape[0])
                ]), label="Target Overlap Save")
                axs[2][1].plot(sum([
                    np.convolve(d_validation_rir[0][l], s_b.y[l])
                    for l in range(room.speaker_coords.shape[0])
                ]), label="Output Overlap Save")
                axs[2][1].legend()

                plt.show()
                    
    # Create output paths (very ugly)
    for i in range(0, Neig, Neig_hop):
        if mode == "apvast":
            name = f"{mode}_{tv}_{float(i)}_{pathlib.Path(b_filename).stem}_{pathlib.Path(d_filename).stem}"
            output_path = pathlib.Path(f"./output/{name}")
        else:
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
                    np.convolve(s_b_t[i // Neig_hop].y[l], b_control_rir[m][l])
                    for l in range(Ns)
                ])
            )
            wavwrite(bright_control_path_output/ f"{m}.wav", 
                sum([
                    np.convolve(s_b[i // Neig_hop].y[l], b_control_rir[m][l])
                    for l in range(Ns)
                ])
            )
            wavwrite(bright_control_path_total/ f"{m}.wav", 
                sum([
                    np.convolve(s_b[i // Neig_hop].y[l] + s_d_t[i // Neig_hop].y[l], b_control_rir[m][l])
                    for l in range(Ns)
                ])
            )
            wavwrite(bright_validation_path_target / f"{m}.wav", 
                sum([
                    np.convolve(s_b_t[i // Neig_hop].y[l], b_validation_rir[m][l])
                    for l in range(Ns)
                ])
            )
            wavwrite(bright_validation_path_output/ f"{m}.wav", 
                sum([
                    np.convolve(s_b[i // Neig_hop].y[l], b_validation_rir[m][l])
                    for l in range(Ns)
                ])
            )
            wavwrite(bright_validation_path_total/ f"{m}.wav", 
                sum([
                    np.convolve(s_b[i // Neig_hop].y[l] + s_d_t[i // Neig_hop].y[l], b_validation_rir[m][l])
                    for l in range(Ns)
                ])
            )

            wavwrite(dark_control_path_target / f"{m}.wav", 
                sum([
                    np.convolve(s_d_t[i // Neig_hop].y[l], d_control_rir[m][l])
                    for l in range(Ns)
                ])
            )
            wavwrite(dark_control_path_output/ f"{m}.wav", 
                sum([
                    np.convolve(s_b[i // Neig_hop].y[l], d_control_rir[m][l])
                    for l in range(Ns)
                ])
            )
            wavwrite(dark_control_path_total/ f"{m}.wav", 
                sum([
                    np.convolve(s_b[i // Neig_hop].y[l] + s_d_t[i // Neig_hop].y[l], d_control_rir[m][l])
                    for l in range(Ns)
                ])
            )
            wavwrite(dark_validation_path_target / f"{m}.wav", 
                sum([
                    np.convolve(s_d_t[i // Neig_hop].y[l], d_validation_rir[m][l])
                    for l in range(Ns)
                ])
            )
            wavwrite(dark_validation_path_output/ f"{m}.wav", 
                sum([
                    np.convolve(s_b[i // Neig_hop].y[l], d_validation_rir[m][l])
                    for l in range(Ns)
                ])
            )
            wavwrite(dark_validation_path_total/ f"{m}.wav", 
                sum([
                    np.convolve(s_b[i // Neig_hop].y[l] + s_d_t[i // Neig_hop].y[l], d_validation_rir[m][l])
                    for l in range(Ns)
                ])
            )
        
    
