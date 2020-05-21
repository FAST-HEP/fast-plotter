import numpy as np
from fast_plotter import plotting


def test_replace_inf():
    x = np.arange(5)
    replaced = plotting.replace_infs(x)
    assert np.array_equal(replaced, x)

    x = np.concatenate(([-np.inf], x, [np.inf]), axis=0)
    replaced = plotting.replace_infs(x)
    assert np.array_equal(replaced, np.arange(-1, 6))

    x = np.arange(2, 4)
    replaced = plotting.replace_infs(x)
    assert np.array_equal(replaced, np.arange(2, 4))

    x = np.concatenate(([-np.inf], x), axis=0)
    replaced = plotting.replace_infs(x)
    assert np.array_equal(replaced, np.arange(1, 4))


def test_pad_zero_noYs():
    x = np.arange(5)
    padded, ticks = plotting.standardize_values(x)
    assert np.array_equal(padded, np.arange(-1, 6))
    assert ticks is None

    x = np.concatenate(([-np.inf], x, [np.inf]), axis=0)
    padded, _ = plotting.standardize_values(x)
    assert np.array_equal(padded, np.arange(-2, 7))

    x = np.arange(2, 4)
    padded, _ = plotting.standardize_values(x)
    assert np.array_equal(padded, np.arange(1, 5))

    x = np.concatenate(([-np.inf], x), axis=0)
    padded, _ = plotting.standardize_values(x)
    assert np.array_equal(padded, np.arange(0, 5, dtype=float))


def test_pad_zero_oneY():
    x = np.arange(5)
    y = np.arange(5, 0, -1)
    expected_y = np.concatenate(([0], y, [0]), axis=0)
    pad_x, ticks, pad_y = plotting.standardize_values(x, [y])
    assert np.array_equal(pad_x, np.arange(-1, 6))
    assert np.array_equal(pad_y, expected_y)
    assert ticks is None

    x = np.concatenate(([-np.inf], x, [np.inf]), axis=0)
    y = np.arange(len(x), 0, -1)
    expected_y = np.concatenate(([0], y, [0]), axis=0)
    pad_x, _, pad_y = plotting.standardize_values(x, [y])
    assert np.array_equal(pad_x, np.arange(-2, 7))
    assert np.array_equal(pad_y, expected_y)

    x = np.arange(2, 4)
    y = np.arange(len(x), 0, -1)
    expected_y = np.concatenate(([0], y, [0]), axis=0)
    pad_x, _, pad_y = plotting.standardize_values(x, [y])
    assert np.array_equal(pad_x, np.arange(1, 5))
    assert np.array_equal(pad_y, expected_y)

    x = np.concatenate(([-np.inf], x), axis=0)
    y = np.arange(len(x), 0, -1)
    expected_y = np.concatenate(([0], y, [0]), axis=0)
    pad_x, _, pad_y = plotting.standardize_values(x, [y])
    assert np.array_equal(pad_x, np.arange(0, 5))
    assert np.array_equal(pad_y, expected_y)


def test_add_missing_vals():
    x = np.arange(3) * 2
    expected = np.arange(7)
    outx, _ = plotting.add_missing_vals(x, expected)
    assert np.array_equal(outx, expected)

    y = np.arange(3)[::-1] + 1
    outx, outy = plotting.add_missing_vals(x, expected, y_values=[y])
    assert np.array_equal(outx, expected)
    assert np.array_equal(outy[0], [3, 0, 2, 0, 1, 0, 0])
    assert outy[0].dtype == y.dtype

    x = np.logspace(0, 10, 11, dtype=int)
    expected = np.zeros(22, dtype=int)
    expected[0::2] = x
    expected[1::2] = x / 2
    y = np.linspace(1, 3, 11)
    outx, (outy,) = plotting.add_missing_vals(x, expected, y_values=[y], fill_val=0)
    assert np.array_equal(outx, expected)
    assert np.array_equal(outy[::2], y)
    assert all(outy[1::2] == 0)
    assert outy.dtype == y.dtype
    assert outx.dtype == expected.dtype
