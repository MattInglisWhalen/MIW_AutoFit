"""
Contains the simple Datum1D class, a container for pointlike 1D data (1 independent variable)
with uncertainties in x and/or y or neither
"""
from dataclasses import dataclass


@dataclass
class Datum1D:
    """
    A container for `pos`, `val`, `sigma_pos`, `sigma_val`, and possibly
    `assym_sigma_pos` or `assym_sigma_val`
    """
    pos: float  # the position of the measurement
    val: float  # the value of the measurement

    sigma_pos: float = 0.0  # the uncertainty of the position
    sigma_val: float = 0.0  # the uncertainty of the value

    assym_sigma_pos: (float, float) = (
        0.0,
        0.0,
    )  # (lower, upper) uncertainties, mostly for histograms in log space
    assym_sigma_val: (float, float) = (
        0.0,
        0.0,
    )  # (lower, upper) uncertainties, not used yet

    def __lt__(self, other):
        return self.pos < other.pos

    def __copy__(self):
        return Datum1D(
            pos=self.pos,
            val=self.val,
            sigma_pos=self.sigma_pos,
            sigma_val=self.sigma_val,
            assym_sigma_pos=self.assym_sigma_pos,
            assym_sigma_val=self.assym_sigma_val,
        )
