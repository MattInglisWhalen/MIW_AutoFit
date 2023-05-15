# built-in packages

# external libraries

# internal classes
import autofit.src.package as pkg

def test_dev_1(capture_stdout):
    pkg.DEV = 1
    pkg._set_package_path()

    assert pkg.pkg_path()[-7:] == "autofit"
    assert pkg.PACKAGE_PATH[-7:] == "autofit"
    pkg.logger("printed")
    assert capture_stdout["stdout"] == f"{pkg.pkg_path()}\nprinted\n"
    with open(pkg.pkg_path()+"/autofit.log") as file:
        line1 = file.readline()
        assert line1 == pkg.pkg_path()+'\n'
        line2 = file.readline()
        assert line2 == ""

def test_dev_0():
    pkg.DEV = 0
    pkg._set_package_path()

    assert pkg.pkg_path()[-7:] == "autofit"
    assert pkg.PACKAGE_PATH[-7:] == "autofit"
    pkg.logger("logged")
    with open(pkg.pkg_path()+"/autofit.log") as file:
        line1 = file.readline()
        assert line1 == pkg.pkg_path()+'\n'
        line2 = file.readline()
        assert line2 == "logged"+'\n'
        line3 = file.readline()
        assert line3 == ""


def test_dev_neg1():
    pkg.DEV = -1
    pkg._set_package_path()

    assert pkg.pkg_path()[-7:] == "autofit"
    assert pkg.PACKAGE_PATH[-7:] == "autofit"
    pkg.logger("/dev/null")
    with open(pkg.pkg_path()+"/autofit.log") as file:
        line1 = file.readline()
        assert line1 == pkg.pkg_path()+'\n'
        line2 = file.readline()
        assert line2 == ""
