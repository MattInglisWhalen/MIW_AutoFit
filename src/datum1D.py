
from dataclasses import dataclass

@dataclass
class Datum1D:

    pos: float              # the position of the measurement
    val: float              # the value of the measurement
    sigma_pos: float = 0.   # the uncertainty of the position
    sigma_val: float = 0.   # the uncertainty of the value

    def __lt__(self, other):
        return self.pos < other.pos
