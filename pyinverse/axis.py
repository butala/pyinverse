import enum
from dataclasses import dataclass

import numpy as np
import scipy.fft


class Order(enum.Enum):
    INCREASING = enum.auto()
    FFT = enum.auto()


# MORE BOUNDS ASSERTIONS LIKE WAS ADDED TO dft METHOD!

# ADD ASSERTIONS THAT AXIS / GRID SHAPES AND ARRAY INPUTS ARE CONSISTENT

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
        self._order = Order.INCREASING

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

    @property
    def N_fast(self):
        """ ??? """
        # This also works for rfft?
        return scipy.fft.next_fast_len(self.N)

    @classmethod
    def linspace(cls, start, stop, num=50, endpoint=True):
        """Return a :class:`RegularAxis` with start point *start*, end point
        *stop*, and *num* number of samples. If *endpoint*, then
        *stop* is the last sample. This function uses the same
        conventions as :func:numpy.linspace (and the Matlab routine by
        the same name).

        """
        x0 = start
        N = num
        if endpoint:
            T = (stop - start) / (num - 1)
        else:
            T =  (stop - start) / num
        return cls(x0, T, N)

    def oversample(self, U):
        """Return a new :class:`RegularAxis` with the same start point as
        *regular* but with a factor *U* more sample points.

        """
        return RegularAxis(self.x0, self.T/U, self.N*U)

    def scale(self, S):
        """ ??? """
        a = RegularAxis(self.x0*S, self.T*S, self.N)
        return a

    def to_Hz(self):
        """ ??? """
        # Need to have units to know if this is valid
        return self.scale(1 / (2*np.pi))

    def spectrum_axis(self, n, real=False):
        """ ??? """
        d = self.T/(2*np.pi)
        if real:
            axis_freq = RFFTRegularAxis(n, self, d=d)
        else:
            axis_freq = FFTRegularAxis(n, self, d=d)
        return axis_freq

    def spectrum(self, x, n=None, real=False):
        """ ??? """
        assert len(x) == self.N
        assert self._order == Order.INCREASING
        if n is None:
            n = self.N
        elif n < self.N:
            return RegularAxis(self.x0, self.T, n).spectrum(x[:n], real=real)
        if real:
            assert np.isrealobj(x)
            X_spectrum = scipy.fft.rfft(x, n=n)
        else:
            X_spectrum = scipy.fft.fft(x, n=n)
        axis_freq = self.spectrum_axis(n, real=real)
        P = np.exp(-1j*axis_freq.centers*self.x0)
        X_spectrum *= P * self.T
        return axis_freq, X_spectrum


class FreqRegularAxis(RegularAxis):
    axis_t: RegularAxis
    _N_FULL: int

    @property
    def d(self):
        """ """
        return 1 / (self.T * self._N_FULL)

    def oversample(self, U):
        """Return a new :class:`RegularAxis` with the same start point as
        *regular* but with a factor *U* more sample points.

        """
        return type(self)(self._N_FULL*U, self.axis_t, d=self.d)

    def scale(self, S):
        """ ??? """
        return type(self)(self._N_FULL, self.axis_t, d=self.d/S)


    def ispectrum_axis(self, n):
        """ ??? """
        return RegularAxis(self.axis_t.x0,
                           2 * np.pi / (self.T * self._N_FULL),
                           self.axis_t.N)

    def ispectrum(self, X_spectrum, n=None, _ifft=None):
        """
        """
        assert len(X_spectrum) == self.N
        if n is not None:
            assert self.axis_t is None
        if n is None:
            n = self._N_FULL
        elif n < self._N_FULL:
            raise NotImplementedError()
        axis_t = self.ispectrum_axis(n)
        P = np.exp(1j*self.centers*axis_t.x0)
        x = _ifft(X_spectrum * P / axis_t.T, n=n)[:axis_t.N]
        return axis_t, x


@dataclass(init=False)
class FFTRegularAxis(FreqRegularAxis):
    def __init__(self, N, axis_t, d=1):
        """Construct regular axis with the same start point and spacing as
        `scipy.fft.fftfreq`.

        Note: *d* is the same default (`d=1`) as
        :func:scipy.fft.fftfreq which results in frequencies ranging
        from -1/2 to 1/2. Set `d=1/(2*np.pi)` to use -pi to pi
        convention.

        """
        if N % 2 == 0:
            x0 = -1/(2*d)
        else:
            x0 = -(N-1)/(2*d*N)
        self._d = d
        super().__init__(x0, 1/(d*N), N)
        self.axis_t = axis_t
        self._N_FULL = N
        self._centers = scipy.fft.fftfreq(self.N, d=self._d)
        self._order = Order.FFT

    def __post_init__(self):
        """
        """
        pass

    def increasing(self, x=None):
        """
        """
        increasing_axis = RegularAxis(self.x0, self.T, self.N)
        if x is None:
            return increasing_axis
        else:
            return increasing_axis, scipy.fft.fftshift(x)

    def ispectrum(self, *args, **kwds):
        """ ??? """
        return super().ispectrum(*args, _ifft=scipy.fft.irfft, **kwds)


@dataclass(init=False)
class RFFTRegularAxis(FreqRegularAxis):
    def __init__(self, N, axis_t, d=1):
        """Construct regular axis with the same start point and spacing as
        `scipy.fft.rfftfreq`.

        Refer to note in :class:FFTRegularAxis

        """
        super().__init__(0, 1/(d*N), N//2+1)
        # Later calls to, e.g., scipy.fft.rfft, require N. However,
        # N//2 + 1 is not an invertible operation and we cannot
        # recover N from N//2 + 1. Save N for later use.
        self._N_FULL = N
        self.axis_t = axis_t

    def ispectrum(self, X_spectrum, **kwds):
        """ ??? """
        assert len(X_spectrum) == self._N_FULL//2 + 1
        return super().ispectrum(X_spectrum, _ifft=scipy.fft.irfft, **kwds)


def dtft(x, real=False, n=None, n0=0):
    """ ??? """
    if n is None:
        n = len(x)
    return RegularAxis(n0, 1, len(x)).spectrum(x, real=real, n=n)


# ADD idft!!!


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
        if self.axis_x._order == Order.INCREASING and self.axis_y._order == Order.INCREASING:
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

    @property
    def N_fast(self):
        """ ??? """
        # This also works for rfft?
        return self.axis_x.N_fast, self.axis_y.N_fast

    def scale(self, Sx, Sy):
        """ ??? """
        return RegularGrid(self.axis_x.scale(Sx), self.axis_y.scale(Sy))

    def to_Hz(self, axis=None):
        """ ??? """
        # Need to have units to know if this is valid
        scale_map = {None: (1 / (2*np.pi), 1 / (2*np.pi)),
                     0:    (1, 1 / (2*np.pi)),
                     1:    (1 / (2*np.pi), 1)}

        #return self.scale(1 / (2*np.pi), 1 / (2*np.pi))
        return self.scale(*scale_map[axis])


    def increasing(self, x=None):
        """ ??? """
        # (axis_x order, axis_y, order) -> ((increasing?, increasing?), axes)
        xform_map = {(Order.FFT, Order.FFT):        ((True, True),  None),
                     (Order.FFT, Order.INCREASING): ((True, False), 1),
                     (Order.INCREASING, Order.FFT): ((False, True), 0),
                     (Order.INCREASING, Order.INCREASING): ((False, False), False)}


        if x is not None:
            assert x.shape == self.shape

        (increasing_x, increasing_y), axes = xform_map[(self.axis_x._order,
                                                        self.axis_y._order)]

        grid = RegularGrid(self.axis_x.increasing() if increasing_x else self.axis_x,
                           self.axis_y.increasing() if increasing_y else self.axis_y)

        if x is not None:
            if axes is not False:
                x = scipy.fft.fftshift(x, axes=axes)
            return grid, x
        else:
            return grid


    def imshow(self, ax, X, interpolation='none', **kwds):
        """Display the 2-D array *X* as an image on axis *ax* taking into
        account the grid orientation. Additional arguments *kwds* are
        passed to :func:`imshow`.

        For details: https://matplotlib.org/3.3.1/tutorials/intermediate/imshow_extent.html

        """
        if self.axis_x._order != Order.INCREASING or self.axis_y._order != Order.INCREASING:
            grid_increasing, X_increasing = self.increasing(X)
            return grid_increasing.imshow(ax, X_increasing, interpolation=interpolation, **kwds)

        assert self.shape == X.shape
        assert 'origin' not in kwds
        kwds['origin'] = 'lower'
        assert 'extent' not in kwds
        x_extent = [self.axis_x.borders[0], self.axis_x.borders[-1]]
        y_extent = sorted([self.axis_y.borders[0], self.axis_y.borders[-1]])
        kwds['extent'] = x_extent + y_extent
        return ax.imshow(X, interpolation=interpolation, **kwds)


    # y = rows, x = cols
    # s order is (y, x)
    def spectrum_grid(self, s=None, axis=None, real=False):
        """ ??? """
        if s is None:
            s = self.shape
        AXIS_MAP = {None: ('F', 'F'),
                    0:    ('F', 'S'),
                    1:    ('S', 'F')}
        if real:
            AXIS_MAP[None] = ('F (real=False)', 'F')

        grid_freq_axis = []
        for xform, n, self_axis in zip(AXIS_MAP[axis], s, [self.axis_y, self.axis_x]):
            if xform == 'F':
                a = self_axis.spectrum_axis(n, real=real)
            elif xform == 'F (real=False)':
                a = self_axis.spectrum_axis(n, real=False)
            elif xform == 'S':
                a = RegularAxis(self_axis.x0, self_axis.T*self_axis.N/n, n)
            else:
                assert False
            grid_freq_axis.append(a)
        return RegularGrid(grid_freq_axis[1], grid_freq_axis[0])


    def spectrum(self, x, s=None, axis=None, real=False):
        """ ??? """
        assert x.shape == self.shape
        assert self.axis_x._order == Order.INCREASING and self.axis_y._order == Order.INCREASING
        if s is None:
            s = self.shape
        elif s < self.shape:
            raise NotImplementedError()
        xform_map = {(True, None):  lambda x, s: scipy.fft.rfft2(x, s=s),
                     (True, 0):     lambda x, s: scipy.fft.rfft(x, n=s[0], axis=0),
                     (True, 1):     lambda x, s: scipy.fft.rfft(x, n=s[1], axis=1),
                     (False, None): lambda x, s: scipy.fft.fft2(x, s=s),
                     (False, 0):    lambda x, s: scipy.fft.fft(x, n=s[0], axis=0),
                     (False, 1):    lambda x, s: scipy.fft.fft(x, n=s[1], axis=1)}
        X_spectrum = xform_map[(real, axis)](x, s)
        grid_freq = self.spectrum_grid(s=s, axis=axis, real=real)
        P = np.exp(-1j*(grid_freq.centers[0]*self.axis_x.x0 + grid_freq.centers[1]*self.axis_y.x0))
        X_spectrum *= P * self.axis_x.T * self.axis_y.T
        return grid_freq, X_spectrum


    def ispectrum_grid(self, s=None, axes=None):
        """ ??? """
        if s is None:
            s = self.shape
        # AXES_MAP = {None: ('F', 'F'),
        #             0:    ('F', 'S'),
        #             1:    ('S', 'F')}
        AXES_MAP = {None: ('S', 'S')}
        # if real:
            # AXES_MAP[None] = ('S', 'S')

        grid_axis = []
        for xform, n, axis in zip(AXES_MAP[axes], s, [self.axis_y, self.axis_x]):
            if xform == 'S':
                if axis._order != Order.FFT:
                    assert isinstance(axis, RFFTRegularAxis)
                a = axis.ispectrum_axis(n)
            else:
                assert False
            grid_axis.append(a)
        return RegularGrid(grid_axis[1], grid_axis[0])


    def ispectrum(self, X_spectrum, s=None, axes=None):
        """ ??? """
        assert X_spectrum.shape == self.shape
        if s is None:
            s = (self.axis_x.axis_t.N, self.axis_y.axis_t.N)
        elif s < self.shape:
            raise NotImplementedError()
        grid = self.ispectrum_grid(s=s, axes=axes)
        P = np.exp(1j*(self.centers[0]*self.axis_x.axis_t._x0 + self.centers[1]*self.axis_y.axis_t.x0))
        assert axes is None
        if isinstance(self.axis_x, FFTRegularAxis) and isinstance(self.axis_y, FFTRegularAxis):
            x = scipy.fft.ifft2(X_spectrum * P / (grid.axis_x.T * grid.axis_y.T), s=s)
        elif isinstance(self.axis_x, RFFTRegularAxis) and isinstance(self.axis_y, FFTRegularAxis):
            x = scipy.fft.irfft2(X_spectrum * P / (grid.axis_x.T * grid.axis_y.T), s=s)
        else:
            assert False
        return grid, x


def dtft2(x, real=False, s=None, n0=(0, 0)):
    """ ??? """
    if s is None:
        s = x.shape
    axis_x = RegularAxis(n0[1], 1, x.shape[1])
    axis_y = RegularAxis(n0[0], 1, x.shape[0])
    grid = RegularGrid(axis_x, axis_y)
    return grid.spectrum(x, real=real, s=s)


# def idtft2(x_dtft2, real=False, s=None, n0=(0, 0)):
#     """ ??? """
#     if s is None:
#         s = x_dtft2.shape
#     axis_x = RegularAxis(n0[1], 1, x_dtft2.shape[1])
#     axis_y = RegularAxis(n0[0], 1, x_dtft2.shape[0])
#     grid = RegularGrid(axis_x, axis_y)
#     grid_spectrum = grid.spectrum_grid(s=s, real=real)
#     return grid_spectrum.ispectrum(x_dtft2, real=real, s=s)
