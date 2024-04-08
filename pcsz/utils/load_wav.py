import scipy as sp
import scipy.io.wavfile
import scipy.signal
import soundfile as sf

def load_wav(filename : str, fs_out, Nx_out, lower_freq=300):
    """Loads a wav file and resamples it appropriately

    Keyword arguments:
    filename -- name of the wav file we wish to load
    fs_out -- sampling rate for output
    Nx_out -- length of signal for output
    """

    # Load from file & convert to mono
    # fs_in, x_in = sp.io.wavfile.read(filename)
    x_in, fs_in = sf.read(filename)
    if x_in.ndim > 1:
        x_in = x_in[:, 0]

    # Apply BPF depending on sampling rate
    sos_bpf = sp.signal.butter(40, [lower_freq, 0.8 * fs_out / 2.0], output='sos', btype='bandpass', fs = fs_in) 
    x_out = sp.signal.sosfilt(sos_bpf, x_in)
    
    # Resample to sampling rate
    assert fs_in % (fs_in // fs_out) == 0, f"For {filename} fs_in = {fs_in} hz not integer multiple of fs_out = {fs_out}"
    x_out = sp.signal.decimate(x_in, fs_in // fs_out)
    
    assert x_out.size >= Nx_out, f"For {filename} resampled length {x_out.size} smaller than desired length {Nx_out}"
    x_out = x_out[0 : Nx_out]

    return x_out
    
