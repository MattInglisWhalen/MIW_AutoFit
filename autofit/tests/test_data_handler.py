"""Tests MIW's AutoFit DataHandler class"""

# required built-ins

# required external libraries
import pytest

# required internal classes
from autofit.src.data_handler import DataHandler
from autofit.src.package import pkg_path
from autofit.tests.conftest import assert_almost_equal


def test_init_scatter():

    with pytest.raises(TypeError):
        # noinspection PyArgumentList
        default = DataHandler()  # pylint: disable=no-value-for-parameter,unused-variable

    csv = DataHandler(filepath=pkg_path() + "/data/linear_data_yerrors.csv")
    xls = DataHandler(filepath=pkg_path() + "/data/file_example_XLS_10.xls")
    xlsx = DataHandler(filepath=pkg_path() + "/data/DampedOscillations.xlsx")
    ods = DataHandler(filepath=pkg_path() + "/data/linear_noerror_multisheet.ods")

    assert csv.shortpath == "linear_data_yerrors.csv"
    assert xls.shortpath == "file_example_XLS_10.xls"
    assert xlsx.shortpath == "DampedOscillations.xlsx"
    assert ods.shortpath == "linear_noerror_multisheet.ods"


def test_init_histo():

    hist = DataHandler(filepath=pkg_path() + "/data/binormal.csv")
    assert hist.shortpath == "binormal.csv"


def test_log_unlog():

    hist = DataHandler(filepath=pkg_path() + "/data/binormal.csv")
    hist.logx_flag = True
    assert hist.logx_flag
    hist.logx_flag = True
    assert hist.logx_flag
    hist.logx_flag = False
    assert not hist.logx_flag
    hist.logx_flag = False
    assert not hist.logx_flag

    hist = DataHandler(filepath=pkg_path() + "/data/random_sample_from_normal_distribution.csv")
    hist.logy_flag = True
    assert hist.logy_flag
    hist.logy_flag = True
    assert hist.logy_flag
    hist.logy_flag = False
    assert not hist.logy_flag
    hist.logy_flag = False
    assert not hist.logy_flag


def test_failed_log():
    # this data is pure-negative in x and y over this excel range
    ods = DataHandler(filepath=pkg_path() + "/data/linear_noerror_multisheet.ods")
    ods.set_excel_args(x_range_str="A3:A18", y_range_str="B3:B18")
    ods.logx_flag = True
    assert not ods.logx_flag
    ods.logx_flag = False
    assert not ods.logx_flag
    ods.logy_flag = True
    assert not ods.logy_flag
    ods.logy_flag = False
    assert not ods.logy_flag


def test_normalize():

    ods = DataHandler(filepath=pkg_path() + "/data/libre_histo.ods")
    ods.set_excel_args(x_range_str="A1:A50")
    assert ods.y_label == "N"
    ods.normalize_histogram_data()
    assert ods.filepath == pkg_path() + "/data/libre_histo.ods"
    assert ods.shortpath == "libre_histo.ods"
    assert ods.x_label == "x"
    assert ods.y_label == "probability density"
    assert_almost_equal(ods.bin_width() * sum((datum.val for datum in ods.data)), 1)
