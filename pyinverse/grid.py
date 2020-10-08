from dataclasses import dataclass
import math

import numpy as np
import scipy.fft


@dataclass
class RegularAxis:
    """Regular, i.e., equally spaced, points on an axis.

    Args:
        x0 (float): starting point
        T (float): distance between points
        N (int): number of points

    """
    x0: float
    T:  float
    N:  int

    def __post_init__(self):
        self._centers = self.x0 + np.arange(self.N, dtype=np.float) * self.T
        self._bounds = self.x0 - self.T/2 + np.arange(self.N + 1, dtype=np.float) * self.T

    @property
    def centers(self):
        """Return sample points."""
        return self._centers

    @property
    def bounds(self):
        """Return locations of edges between sample points."""
        return self._bounds

    def __iter__(self):
        """Return iterator over sample points."""
        return iter(self.centers)

    def __getitem__(self, k):
        """Return the kth sample point."""
        return self.centers[k]

    def __len__(self):
        """Return the number of sample points."""
        return self.N

    @property
    def extent(self):
        """Return the axis bounds, i.e., the tuple of left edge of the start
point *x0* and the right edge of the last point."""
        return self.bounds[0], self.bounds[-1]


@dataclass(init=False)
class FFTRegularAxis(RegularAxis):
    def __init__(self, N, d=1/(2*math.pi)):
        """Construct regular axis with the same start point and spacing as
        `scipy.fft.fftfreq`.

        Note: *d* is selected such that the axis sample points will
        range from -pi to pi. This is different than `d=1` which is
        the default in `scipy.fft.fftfreq` which results in sample
        points ranging from -1/2 to 1/2.

        """
        if N % 2 == 0:
            x0 = -1/(2*d)
        else:
            x0 = -(N-1)/(2*d*N)
        super().__init__(x0, 1/(d*N), N)


@dataclass(init=False)
class RFFTRegularAxis(RegularAxis):
    def __init__(self, N, d=1/(2*math.pi)):
        """Construct regular axis with the same start point and spacing as
        `scipy.fft.rfftfreq`.

        Refer to note in :class:FFTRegularAxis

        """
        super().__init__(0, 1/(d*N), N//2+1)


@dataclass
class RegularGrid:
    """Regular, i.e., equally spaced, points on a grid.

    Args:
        axis_x (RegularAxis): horizontal axis
        axis_y (RegularAxis): vertical axis

    """
    axis_x: RegularAxis
    axis_y: RegularAxis

    @property
    def shape(self):
        """Return the tuple with the number of vertical and horizontal sample points."""
        return self.axis_y.N, self.axis_x.N

    @property
    def extent(self):
        """Return the grid extent in bounding box format (see :func:`pyinverse.ellipse.get_ellipse_bb`)."""
        return self.axis_x.extent + self.axis_y.extent


def oversample_regular_axis(regular, oversample):
    """Return a new :class:`RegularAxis` with the same start point as
    *regular* but with a factor *oversample* more sample points.

    """
    return RegularAxis(regular.x0, regular.T/oversample, regular.N*oversample)


def oversample_regular_grid(regular_grid, oversample):
    """Return a new :class:`RegularGrid` with each axis having the same
    start point as the corresponding axis in *regular_grid* but with a
    factor *oversample* more sample points.

    """
    return RegularGrid(oversample_regular_axis(regular_grid.axis_x, oversample),
                       oversample_regular_axis(regular_grid.axis_y, oversample))
