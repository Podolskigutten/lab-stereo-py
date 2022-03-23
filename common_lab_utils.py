import numpy as np
from dataclasses import dataclass

class Size:
    """Represents image size"""

    def __init__(self, width: float, height: float):
        self._width = width
        self._height = height

    @classmethod
    def from_numpy_shape(cls, shape):
        return cls(*shape[1::-1])

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height
    
    @property
    def as_cv_size(self):
        return np.array((self._width, self._height), dtype=int)

@dataclass
class StereoPair:
    left : np.ndarray
    right : np.ndarray