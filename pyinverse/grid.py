from dataclasses import dataclass
import math

import numpy as np
import scipy.fft


# NEED TO ADD ABILITY FOR USER TO GIVE ZERO PADDING AMOUNT --- AND THEN ZERO PAD ON TOP TO MAKE IT FAST!

# ADD UNIT (default of s of axis, m for grid, [length] for phantom --- can include this on axis labels in imshow)
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

    def dft_axis(self, real=False, zero_pad=True):
        """
        """
        if zero_pad:
            # This also works for rfft?
            N_fast = scipy.fft.next_fast_len(self.N)
        else:
            N_fast = self.N
        if real:
            omega_axis = RFFTRegularAxis(N_fast, d=1/(2*math.pi))
        else:
            omega_axis = FFTRegularAxis(N_fast, d=1/(2*math.pi))
        omega_axis._axis_t = self
        omega_axis._real = real
        return omega_axis

    def dft(self, x, real=False, zero_pad=True, **kwds):
        """
        """
        if real:
            assert np.isrealobj(x)
        omega_axis = self.dft_axis(real=real, zero_pad=zero_pad)
        if real:
            N_fast = omega_axis._N_FULL
            X_DFT = scipy.fft.rfft(x, n=N_fast, **kwds)
        else:
            N_fast = omega_axis.N
            X_DFT = scipy.fft.fftshift(scipy.fft.fft(x, n=N_fast, **kwds))
        return omega_axis, X_DFT

    def ft_axis(self, real=False, zero_pad=True, _dft_axis=None):
        """
        """
        if _dft_axis is None:
            _dft_axis = self.dft_axis(real=real, zero_pad=zero_pad)

        if real:
            axis_f = RFFTRegularAxis(_dft_axis._N_FULL, d=self.T)
        else:
            axis_f = FFTRegularAxis(_dft_axis.N, d=self.T)
        axis_f._axis_t = self
        axis_f._real = real
        return axis_f

    def ft(self, x, real=False, zero_pad=True, **kwds):
        """
        """
        omega_axis, X_DFT = self.dft(x, real=real, zero_pad=zero_pad, **kwds)
        axis_f = self.ft_axis(real=real, zero_pad=zero_pad, _dft_axis=omega_axis)
        X_FT = X_DFT*self.T * np.exp(-1j*2*np.pi*self[0]*axis_f.centers)
        return axis_f, X_FT

    def idft(self, X_dft, **kwds):
        """
        """
        try:
            N = self._axis_t.N
        except AttributeError:
            N = self.N
        if self._real:
            # EXPLAIN WHY n=N here!
            x = scipy.fft.irfft(X_dft, n=N, **kwds)
        else:
            x = scipy.fft.ifft(scipy.fft.ifftshift(X_dft), **kwds)
        n_axis = RegularAxis(0, 1, N)
        return n_axis, x[:N]

    def ift(self, X_ft, **kwds):
        """
        """
        if self._real:
            axis_t_T = (1/self.T)/(2*(self.N-1))
        else:
            axis_t_T = (1/self.T)/self.N
        X_dft = X_ft/axis_t_T * np.exp(1j*2*np.pi*self._axis_t[0]*self.centers)
        n_axis, x = self.idft(X_dft, **kwds)
        axis_t = FFTRegularAxis(n_axis.N, d=1/(axis_t_T * n_axis.N))
        return axis_t, x


@dataclass(init=False)
class FFTRegularAxis(RegularAxis):
    def __init__(self, N, d=1):
        """Construct regular axis with the same start point and spacing as
        `scipy.fft.fftfreq`.

        Note: *d* is the same default (`d=1`) as
        :func:scipy.fft.fftfreq which results in frequencies ranging
        from -1/2 to 1/2. Set `d=1/(2*math.pi)` to use -pi to pi
        convention.

        """
        if N % 2 == 0:
            x0 = -1/(2*d)
        else:
            x0 = -(N-1)/(2*d*N)
        super().__init__(x0, 1/(d*N), N)

@dataclass(init=False)
class RFFTRegularAxis(RegularAxis):
    def __init__(self, N, d=1):
        """Construct regular axis with the same start point and spacing as
        `scipy.fft.rfftfreq`.

        Refer to note in :class:FFTRegularAxis

        """
        super().__init__(0, 1/(d*N), N//2+1)
        # Later calls to, e.g., scipy.fft.rfft, require N. However,
        # N//2 + 1 is not an invertible operation and we cannot
        # recover N from N//2 + 1. Save N for later use.
        self._N_FULL = N

@dataclass
class RegularGrid:
    """Regular, i.e., equally spaced, points on a grid.

    Args:
        axis_x (RegularAxis): horizontal axis
        axis_y (RegularAxis): vertical axis

    """
    axis_x: RegularAxis
    axis_y: RegularAxis

    def __post_init__(self):
        # For reference on all the numpy grid functions: https://stackoverflow.com/questions/12402045/mesh-grid-functions-in-python-meshgrid-mgrid-ogrid-ndgrid
        self._centers = np.meshgrid(self.axis_x.centers, self.axis_y.centers)
        self._borders = np.meshgrid(self.axis_x.borders, self.axis_y.borders)

    @property
    def centers(self):
        """ ??? """
        return self._centers

    @property
    def borders(self):
        """ ??? """
        return self._borders

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

    def dft_grid(self, axis=None, real=False, zero_pad=True):
        """
        ???
        """
        if axis is None:
            axis_fx = self.axis_x.dft_axis(real=real, zero_pad=zero_pad)
            # EXPLAIN real=False
            axis_fy = self.axis_y.dft_axis(real=False, zero_pad=zero_pad)
            return RegularGrid(axis_fx, axis_fy)
        elif axis == 0:
            axis_fy = self.axis_y.dft_axis(real=real, zero_pad=zero_pad)
            return RegularGrid(self.axis_x, axis_fy)
        elif axis == 1:
            axis_fx = self.axis_x.dft_axis(real=real, zero_pad=zero_pad)
            return RegularGrid(axis_fx, self.axis_y)
        else:
            raise ValueError(f'unknown axis {axis}')

    def dft(self, x, axis=None, real=False, zero_pad=True, **kwds):
        """ ??? """
        # NEED COMMENTS TO EXPLAIN TREE OF CASES BELOW
        grid_dft = self.dft_grid(axis=axis, real=real, zero_pad=zero_pad)
        if axis is None:
            if real:
                s = [grid_dft.axis_y.N, grid_dft.axis_x._N_FULL]
                X = scipy.fft.fftshift(scipy.fft.rfft2(x, s=s, **kwds), axes=0)
            else:
                s = [grid_dft.axis_y.N, grid_dft.axis_x.N]
                X = scipy.fft.fftshift(scipy.fft.fft2(x, s=s, **kwds))
            return grid_dft, X
        elif axis == 0:
            if real:
                X = scipy.fft.rfft(x, n=grid_dft.axis_y._N_FULL, axis=0, **kwds)
            else:
                X = scipy.fft.fftshift(scipy.fft.fft(x, n=grid_dft.axis_y.N, axis=0, **kwds), axes=0)
            return grid_dft, X
        elif axis == 1:
            if real:
                X = scipy.fft.rfft(x, n=grid_dft.axis_x._N_FULL, axis=1, **kwds)
            else:
                X = scipy.fft.fftshift(scipy.fft.fft(x, n=grid_dft.axis_x.N, axis=1, **kwds), axes=1)
            return grid_dft, X
        else:
            raise ValueError(f'unknown axis {axis}')

    def ft_grid(self, axis=None, real=False, zero_pad=True, _dft_grid=None):
        """
        """
        if _dft_grid is None:
            _dft_grid = self.dft_grid(axis=axis, real=real, zero_pad=zero_pad)
        if axis is None:
            return RegularGrid(
                self.axis_x.ft_axis(real=real,
                                    zero_pad=zero_pad,
                                    _dft_axis=_dft_grid.axis_x),
                self.axis_y.ft_axis(real=False,
                                    zero_pad=zero_pad,
                                    _dft_axis=_dft_grid.axis_y))
        elif axis == 0:
            axis_fy = self.axis_y.ft_axis(real=real, zero_pad=zero_pad, _dft_axis=_dft_grid.axis_y)
            return RegularGrid(self.axis_x, axis_fy)
        elif axis == 1:
            axis_fx = self.axis_x.ft_axis(real=real, zero_pad=zero_pad, _dft_axis=_dft_grid.axis_x)
            return RegularGrid(axis_fx, self.axis_y)
        else:
            raise ValueError(f'unknown axis {axis}')

    def ft(self, x, axis=None, real=False, zero_pad=True, **kwds):
        """ ??? """
        # NEED COMMENTS TO EXPLAIN TREE OF CASES BELOW
        if axis is None:
            omega_grid, X_DFT2 = self.dft(x, axes=None, real=real, zero_pad=zero_pad, **kwds)
            f_grid = self.ft_grid(axis=axis, real=real, zero_pad=zero_pad, _dft_grid=omega_grid)
            P = np.exp(-1j*2*np.pi*(self.axis_x[0]*f_grid.centers[0] + self.axis_y[0]*f_grid.centers[1]))
            X_FT2 = X_DFT2 * self.axis_x.T * self.axis_y.T * P
            return f_grid, X_FT2
        elif axis == 0:
            omega_grid, X_DFT = self.dft(x, axis=0, real=real, zero_pad=zero_pad, **kwds)
            axis_fy = self.axis_y.ft_axis(real=real, zero_pad=zero_pad, _dft_axis=omega_grid.axis_y)

            p = np.exp(-1j*2*np.pi*self.axis_y[0]*axis_fy.centers)
            X_FT = X_DFT*self.axis_y.T * np.atleast_2d(p).T
            return RegularGrid(self.axis_x, axis_fy), X_FT
        elif axis == 1:
            omega_grid, X_DFT = self.dft(x, axis=1, real=real, zero_pad=zero_pad, **kwds)
            axis_fx = self.axis_x.ft_axis(real=real, zero_pad=zero_pad, _dft_axis=omega_grid.axis_x)

            p = np.exp(-1j*2*np.pi*self.axis_x[0]*axis_fx.centers)
            X_FT = X_DFT*self.axis_x.T * np.atleast_2d(p)
            return RegularGrid(axis_fx, self.axis_y), X_FT
        else:
            raise ValueError(f'unknown axis {axis}')

    @property
    def _real(self):
        """
        """
        if self.axis_x._real:
            assert not self.axis_y._real
            return True
        else:
            return False

    def idft(self, X_dft, axis=None, **kwds):
        """
        """
        if axis is None:
            try:
                Nx = self.axis_x._axis_t.N
            except AttributeError:
                Nx = self.axis_x.N
            try:
                Ny = self.axis_y._axis_t.N
            except AttributeError:
                Ny = self.axis_y.N
            if self._real:
                # Explain why s= here!!!
                x = scipy.fft.irfft2(scipy.fft.ifftshift(X_dft, axes=0), s=(Ny, Nx), **kwds)
            else:
                x = scipy.fft.ifft2(scipy.fft.ifftshift(X_dft), **kwds)
            n_grid = RegularGrid(RegularAxis(0, 1, Nx), RegularAxis(0, 1, Ny))
            return n_grid, x[:Ny, :Nx]
        elif axis == 0:
            assert False
        elif axis == 1:
            assert False
        else:
            raise ValueError(f'unknown axis {axis}')

    def ift(self, X_ft, axis=None, **kwds):
        """
        """
        if axis is None:
            if self._real:
                axis_tx_T = (1/self.axis_x.T)/(2*(self.axis_x.N-1))
            else:
                axis_tx_T = (1/self.axis_x.T)/self.axis_x.N
            axis_ty_T = (1/self.axis_y.T)/self.axis_y.N
            P = np.exp(1j*2*np.pi*(self.axis_x._axis_t[0]*self.centers[0] + self.axis_y._axis_t[0]*self.centers[1]))
            X_dft = X_ft/axis_tx_T/axis_ty_T * P
            n_grid, x = self.idft(X_dft, axis=None, **kwds)

            grid = RegularGrid(FFTRegularAxis(n_grid.axis_x.N, d=1/(axis_tx_T * n_grid.axis_x.N)),
                               FFTRegularAxis(n_grid.axis_y.N, d=1/(axis_ty_T * n_grid.axis_y.N)))
            return grid, x
        elif axis == 0:
            assert False
        elif axis == 1:
            assert False
        else:
            raise ValueError(f'unknown axis {axis}')

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
