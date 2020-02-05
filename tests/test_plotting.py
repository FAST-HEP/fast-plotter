import numpy as np
from fast_plotter import plotting


def test_pad_zero_noYs():
    x = np.arange(5)
    padded = plotting.pad_zero(x)
    assert (padded == np.arange(-1, 6)).all()

    x = np.concatenate(([-np.inf], x, [np.inf]), axis=0)
    padded = plotting.pad_zero(x)
    assert (padded == np.arange(-1, 6)).all()

    x = np.arange(2, 4)
    padded = plotting.pad_zero(x)
    assert (padded == np.arange(1, 5)).all()

    x = np.concatenate(([-np.inf], x), axis=0)
    padded = plotting.pad_zero(x)
    assert (padded == np.arange(1, 5)).all()

def test_pad_zero_oneY():

    x = np.arange(5)
    y = np.arange(5, 0, -1)
    pad_x, pad_y = plotting.pad_zero(x, [y])
    assert (pad_x == np.arange(-1, 6)).all()
    expected_y = np.concatenate(([0], y, [0]), axis=0)
    assert np.array_equal(pad_y, expected_y)

    x = np.concatenate(([-np.inf], x, [np.inf]), axis=0)
    y = np.arange(len(x), 0, -1)
    pad_x, pad_y = plotting.pad_zero(x, [y])
    assert (pad_x == np.arange(-1, 6)).all()
    assert np.array_equal(pad_y, y)

    x = np.arange(2, 4)
    y = np.arange(len(x), 0, -1)
    pad_x, pad_y = plotting.pad_zero(x, y)
    print(x, y)
    print(pad_x, pad_y)
    assert (pad_x == np.arange(1, 5)).all()
    expected_y = np.concatenate(([0], y, [0]), axis=0)
    assert np.array_equal(pad_y, expected_y)

    x = np.concatenate(([-np.inf], x), axis=0)
    y = np.arange(len(x), 0, -1)
    pad_x, pad_y = plotting.pad_zero(x, y)
    print(x, y)
    print(pad_x, pad_y)
    assert (pad_x == np.arange(1, 5)).all()
    expected_y = np.concatenate((y, [0]), axis=0)
    assert np.array_equal(pad_y, expected_y)

    assert False
