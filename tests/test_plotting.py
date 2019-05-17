import numpy as np
from fast_plotter import plotting


def test_pad_zero():
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
