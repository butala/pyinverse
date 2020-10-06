from dataclasses import dataclass
import math

import numpy as np
import scipy.fft


@dataclass
class Regular:
    x0: float
    T:  float
    N:  int

    @property
    def centers(self):
        try:
            return self._centers
        except AttributeError:
            self._centers = self.x0 + np.arange(self.N, dtype=np.float) * self.T
            return self._centers

    @property
    def bounds(self):
        try:
            return self._bounds
        except AttributeError:
            self._bounds = self.x0 - self.T/2 + np.arange(self.N + 1, dtype=np.float) * self.T
            return self._bounds

    def __iter__(self):
        return iter(self.centers)

    def __getitem__(self, key):
        return self.centers[key]

    @property
    def extent(self):
        return self.bounds[0], self.bounds[-1]


@dataclass(init=False)
class FFTRegular(Regular):
    def __init__(self, N, d=1/(2*math.pi)):
        if N % 2 == 0:
            x0 = -1/(2*d)
        else:
            x0 = -(N-1)/(2*d*N)
        super().__init__(x0, 1/(d*N), N)


@dataclass(init=False)
class RFFTRegular(Regular):
    def __init__(self, N, d=1/(2*math.pi)):
        super().__init__(0, 1/(d*N), N//2+1)


@dataclass
class RegularGrid:
    axis_x: Regular
    axis_y: Regular

    @property
    def shape(self):
        return self.axis_y.N, self.axis_x.N

    @property
    def extent(self):
        return self.axis_x.extent + self.axis_y.extent


def oversample_regular(regular, oversample):
    return Regular(regular.x0, regular.T/oversample, regular.N*oversample)


def oversample_regular_grid(regular_grid, oversample):
    return RegularGrid(oversample_regular(regular_grid.axis_x, oversample),
                       oversample_regular(regular_grid.axis_y, oversample))
