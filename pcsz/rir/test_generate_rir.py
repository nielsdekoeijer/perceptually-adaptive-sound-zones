import pytest

from pcsz.rir.generate_rir import generate_rir

def test_generate_rir():
    Nh                  = 100
    fs                  = 800
    point_coords        = [[1., 1., 1.], [2., 2., 1.], [3., 3., 1.]]
    speaker_coords      = [[0., 1., 1,], [0., 2., 1.]]
    room_dim            = [6, 4, 3]
    reverberation_time  = 0.200
    reflection_order    = 5

    h_test = generate_rir(Nh, fs, point_coords, speaker_coords, room_dim, reverberation_time, reflection_order)
    assert h_test.shape == (len(point_coords), len(speaker_coords), Nh)