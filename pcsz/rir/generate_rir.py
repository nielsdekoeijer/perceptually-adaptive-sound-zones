import rir_generator as rg
import numpy as np

import scipy as sp
import scipy.io

def generate_rir(Nh : int, fs : int, point_coords, speaker_coords, room_dim, \
        reverberation_time, reflection_order):
    """Generate a room impulse response.

    Keyword arguments:
    Nh -- desired lenght of RIRs
    fs -- sampling rare for RIRs
    point_coords -- array of coordinates [[rx1,ry1,rz1], ...] for RIRs 
    speaker_coords -- array of coordinates [[sx1,sy1,sz1], ...] for loudspeakers
    room_dim -- dimension of room [Lx,Ly,Lz]
    reverberation_time -- time in seconds of reverberation for RIRs
    reflection_order -- how many reflection should be simulated
    """

    # Extract number of points
    Np = len(point_coords)

    # Extract number of loudspeakers
    Nl = len(speaker_coords)

    # Pre-allocate RIR vector
    h = np.zeros((Nl, Nh, Np))

    # Generate RIR for each loudspeaker
    for l in range(Nl):
        h[l] = rg.generate(
                c=343, # Speed of sound
                fs=fs, # Sampling rate
                r=point_coords, # Microphone locations
                s=speaker_coords[l], # Speaker positions
                L=room_dim, # Dimension of the room
                nsample=Nh, # Number of RIRs
                reverberation_time=reverberation_time, # RIR reverberation time
                order=reflection_order, # Order of the resulting reflections
            )


    # Transpose to [microphone, loudspeaker, rir] format and return
    return np.swapaxes(np.swapaxes(h, 1, 2), 0, 1)
