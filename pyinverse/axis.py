import enum
from dataclasses import dataclass

import numpy as np
import scipy.fft


class Order(enum.Enum):
    INCREASING = enum.auto()
    FFT = enum.auto()


# MORE BOUNDS ASSERTIONS LIKE WAS ADDED TO dft METHOD!

# NEED TO ADD ABILITY FOR USER TO GIVE ZERO PADDING AMOUNT --- AND THEN ZERO PAD ON TOP TO MAKE IT FAST!

# SHOULD AN AXIS/GRID HAVE A real (True by default) PROPERTY?

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

    def spectrum_axis(self, n, real=False):
        """ ??? """
        d = self.T/(2*np.pi)
        if real:
            axis_freq = RFFTRegularAxis(n, d=d)
        else:
            axis_freq = FFTRegularAxis(n, d=d)
        axis_freq._axis_t_x0 = self.x0
        axis_freq._axis_t_N = self.N
        return axis_freq

    def spectrum(self, x, n=None, real=False):
        """ ??? """
        assert self.N == len(x)
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
        axis_freq._axis_t_x0 = self.x0
        axis_freq._axis_t_N = self.N
        return axis_freq, X_spectrum


class FreqRegularAxis(RegularAxis):
    _axis_t_x0: float = None
    _axis_t_N: int = None
    _N_FULL: int = None

    def ispectrum_axis(self, n):
        """ ??? """
        axis_t_x0 = 0 if self._axis_t_x0 is None else self._axis_t_x0
        axis_t_N = n if self._axis_t_N is None else self._axis_t_N
        return RegularAxis(axis_t_x0,
                           2 * np.pi / (self.T * self._N_FULL),
                           axis_t_N)

    def ispectrum(self, X_spectrum, n=None, _ifft=None):
        """
        """
        assert len(X_spectrum) == self.N
        if n is not None:
            assert self._axis_t_N is None
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
    def __init__(self, N, d=1):
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
        self._N_FULL = N

    def __post_init__(self):
        """ ??? """
        self._centers = scipy.fft.fftfreq(self.N, d=self._d)
        self._order = Order.FFT

    def incresing(self, x):
        """
        """
        increasing_axis = RegularAxis(self.x0, self.T, self.N)
        increasing_axis._axis_t_x0 = self._axis_t_x0
        increasing_axis._axis_t_N = self._axis_t_N
        return increasing_axis, scipy.fft.ffttshift(x)

    def ispectrum(self, *args, **kwds):
        """ ??? """
        return super().ispectrum(*args, _ifft=scipy.fft.irfft, **kwds)


@dataclass(init=False)
class RFFTRegularAxis(FreqRegularAxis):
    _N_FULL: int = None

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

    def ispectrum(self, X_spectrum, **kwds):
        """ ??? """
        assert len(X_spectrum) == self._N_FULL//2 + 1
        return super().ispectrum(X_spectrum, _ifft=scipy.fft.irfft, **kwds)


def dtft(x, real=False, n=None, n0=0):
    """ ??? """
    if n is None:
        n = len(x)
    return RegularAxis(n0, 1, len(x)).spectrum(x, real=real, n=n)
