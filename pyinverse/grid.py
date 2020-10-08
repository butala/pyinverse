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
        self._borders = self.x0 - self.T/2 + np.arange(self.N + 1, dtype=np.float) * self.T

    @property
    def centers(self):
        """Return sample points."""
        return self._centers

    @property
    def borders(self):
        """Return locations of edges between sample points."""
        return self._borders

    def __iter__(self):
        """Return iterator over sample points."""
        return iter(self.centers)

    def __getitem__(self, k):
        """Return the kth sample point."""
        return self.centers[k]

    def __len__(self):
        """Return the number of sample points."""
        return self.N


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

    def __iter__(self):
        """Return an iterator over the x axis and y axis (in that order)."""
        return (x for x in [self.axis_x, self.axis_y])

    @property
    def shape(self):
        """Return the tuple with the number of vertical and horizontal sample points."""
        return self.axis_y.N, self.axis_x.N

    def imshow(self, ax, X, interpolation='none', **kwds):
        """Display the 2-D array *X* as an image on axis *ax* taking into
        account the grid orientation. Additional arguments *kwds* are
        passed to :func:`imshow`.

        For details: https://matplotlib.org/3.3.1/tutorials/intermediate/imshow_extent.html

        """
        if self.axis_x.borders[0] > self.axis_x.borders[-1]:
            # did not implement case with flipped horizontal axis as it would be pretty non-standard --- we could do it, but it will never show up in practice
            raise NotImplementedError('flipped horizontal axis (decreasing from left to right)')
        if self.axis_y.borders[0] > self.axis_y.borders[-1]:
            # row 0 of X corresponds to the smallest y coordinate
            origin = 'upper'
        else:
            # row 0 of X corresponds to the largest y coordinate
            origin = 'lower'
        assert self.shape == X.shape
        assert 'origin' not in kwds
        kwds['origin'] = origin
        assert 'extent' not in kwds
        x_extent = [self.axis_x.borders[0], self.axis_x.borders[-1]]
        y_extent = sorted([self.axis_y.borders[0], self.axis_y.borders[-1]])
        kwds['extent'] = x_extent + y_extent
        return ax.imshow(X, interpolation=interpolation, **kwds)


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
