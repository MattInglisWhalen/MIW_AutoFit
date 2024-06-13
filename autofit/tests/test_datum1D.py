# built-in packages

# external libraries
import pytest

# internal classes
from autofit.src.datum1D import Datum1D


def test_default():
    with pytest.raises(TypeError):
        data = Datum1D()


def test_no_errors():
    data = Datum1D(pos=5.4, val=7.9)

    assert data.pos == 5.4
    assert data.val == 7.9

    assert data.sigma_pos == 0
    assert data.sigma_val == 0

    assert data.assym_sigma_pos == (0, 0)
    assert data.assym_sigma_val == (0, 0)


def test_symm_errors():
    data = Datum1D(pos=5.4, val=7.9, sigma_pos=1.1, sigma_val=1.2)

    assert data.pos == 5.4
    assert data.val == 7.9

    assert data.sigma_pos == 1.1
    assert data.sigma_val == 1.2

    assert data.assym_sigma_pos == (0, 0)
    assert data.assym_sigma_val == (0, 0)


def test_asymm_errors():
    data = Datum1D(
        pos=5.4, val=7.9, assym_sigma_pos=(1.1, 0.2), assym_sigma_val=(1.2, 0.3)
    )

    assert data.pos == 5.4
    assert data.val == 7.9

    assert data.sigma_pos == 0
    assert data.sigma_val == 0

    assert data.assym_sigma_pos == (1.1, 0.2)
    assert data.assym_sigma_val == (1.2, 0.3)


def test_copy():
    data = Datum1D(pos=5.4, val=7.9, sigma_pos=1.1, sigma_val=1.2)
    data_copied = data.__copy__()
    assert data_copied is not data

    assert data_copied.pos == 5.4
    assert data_copied.val == 7.9

    assert data_copied.sigma_pos == 1.1
    assert data_copied.sigma_val == 1.2

    assert data_copied.assym_sigma_pos == (0, 0)
    assert data_copied.assym_sigma_val == (0, 0)


def test_ordering():

    data1 = Datum1D(pos=5.4, val=7.9, sigma_pos=1.1, sigma_val=1.2)
    data2 = Datum1D(pos=0.4, val=7.9, sigma_pos=1.1, sigma_val=1.2)
    data3 = Datum1D(pos=6.4, val=7.9, sigma_pos=1.1, sigma_val=1.2)
    data4 = Datum1D(pos=9.4, val=7.9, sigma_pos=1.1, sigma_val=1.2)
    data5 = Datum1D(pos=-3.4, val=7.9, sigma_pos=1.1, sigma_val=1.2)
    unordered = [data1, data2, data3, data4, data5]

    ordered = sorted(unordered)
    assert ordered[0] is data5
    assert ordered[1] is data2
    assert ordered[2] is data1
    assert ordered[3] is data3
    assert ordered[4] is data4
