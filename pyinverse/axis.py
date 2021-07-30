import enum
from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.fft


class Order(enum.Enum):
    INCREASING = enum.auto()
    FFT = enum.auto()


# ADD UNIT (default of s of axis, m for grid, [length] for phantom --- can include this on axis labels in imshow)
class RegularAxis:
    """Regular, i.e., equally spaced, points on an axis.

    Args:
        x0 (float): starting point
        T (float): distance between points
        N (int): number of points

    """
    def __init__(self, x0, T, N):
        """ ??? """
        self.x0 = x0
        self.T = T
        self.N = N
        self.__post_init__()

    def __post_init__(self):
        self._centers = self.x0 + np.arange(self.N, dtype=np.float) * self.T
        self._borders = self.x0 - self.T/2 + np.arange(self.N + 1, dtype=np.float) * self.T
        self._order = Order.INCREASING

    def __repr__(self):
        return f'<{self.__class__.__name__} x0={self.x0} T={self.T} N={self.N}>'

    def __str__(self):
        return f'{self.__class__.__name__}: {self.centers[0]} -- {self.centers[-1]}, N={self.N}, T={self.T}'

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
        if endpoint and num > 1:
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
        return RegularAxis(self.x0*S, self.T*S, self.N)

    def Hz(self):
        """ ??? """
        # Need to have units to determine if this is valid!
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


class FreqRegularAxis(RegularAxis, metaclass=ABCMeta):
    @property
    @abstractmethod
    def axis_t(self):
        pass

    @property
    @abstractmethod
    def _N_FULL(self):
        pass

    @property
    def d(self):
        """ """
        return 1 / (self.T * self._N_FULL)

    def oversample(self, U):
        """Return a new :class:`FreqRegularAxis` with the same start point as
        *regular* but with a factor *U* more sample points.

        """
        return type(self)(self._N_FULL*U, self.axis_t, d=self.d)

    def scale(self, S):
        """ ??? """
        return type(self)(self._N_FULL, self.axis_t, d=self.d/S)


    def ispectrum(self, X_spectrum, _ifft=None):
        """
        """
        # After some deliberation, the option to override the length
        # parameter n passed to ifft (or irfft) has been
        # removed. Keeping it makes sense to preserve symmetry: time
        # domain zero padding yielding increased frequency domain
        # sampling is conceptually the same as frequency domain zero
        # padding yielding increased time domain sampling.
        #
        # There are several practical issues. First, ifft zero padding
        # is just that, padding the end of the input sequence without
        # regard to fftfreq ordering (0 to highest frequency, lowest
        # to 0) which requires "stuffing" zeros in the middle of the
        # input sequence. Second, proper "stuffing" depends on whether
        # the padded length is even or odd and involves halving
        # X_spectrum[N/2] if N is even. Finally, performance will not
        # be good when the original sequence x[n] from which
        # X_spectrum is determined has significant spectral content at
        # +-fs/2 (Nyquist frequency). All of this is detailed in the
        # following references:
        #
        # Lyons, Understanding Digital Signal Processing, 3rd edition,
        # 2010, section 13.28.
        #
        # Fraser, D. “Interpolation by the FFT Revisited—An
        # Experimental Investigation,” IEEE Trans. on Acoust. Speech,
        # and Signal Proc., Vol. ASSP-37, No. 5, May 1989,
        # pp. 665–676.
        #
        # Given all of these caveats, spatial domain interpolation via
        # frequency domain "zero stuffing" should be implemented as a
        # separate function. The necessary steps are not transparent
        # like zero padding the fft/rfft and presenting it as being
        # similarly straightforward here would be misleading.

        assert len(X_spectrum) == self.N
        P = np.exp(1j*self.centers*self.axis_t.x0)
        x = _ifft(X_spectrum * P / self.axis_t.T, n=self._N_FULL)[:self.axis_t.N]
        return self.axis_t, x


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
        self._axis_t = axis_t
        self.__N_FULL = N
        self._centers = scipy.fft.fftfreq(self.N, d=self._d)
        self._order = Order.FFT

    def __post_init__(self):
        # Override RegularAxis.__post_init__. Everything is taken care
        # of in FFTRegularAxis.__init__ and we do want, for example, a
        # _borders attribute which is set in
        # RegularAxis.__post__init__.
        pass

    @property
    def axis_t(self):
        return self._axis_t

    @property
    def _N_FULL(self):
        return self.__N_FULL

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


class RFFTRegularAxis(FreqRegularAxis):
    def __init__(self, N, axis_t, d=1):
        """Construct regular axis with the same start point and spacing as
        `scipy.fft.rfftfreq`.

        Refer to note in :class:FFTRegularAxis

        """
        super().__init__(0, 1/(d*N), N//2+1)
        self._axis_t = axis_t
        # Later calls to, e.g., scipy.fft.rfft, require N. However,
        # N//2 + 1 is not an invertible operation and we cannot
        # recover N from N//2 + 1. Save N for later use.
        self.__N_FULL = N

    @property
    def axis_t(self):
        return self._axis_t

    @property
    def _N_FULL(self):
        return self.__N_FULL

    def ispectrum(self, X_spectrum, **kwds):
        """ ??? """
        assert len(X_spectrum) == self._N_FULL//2 + 1
        return super().ispectrum(X_spectrum, _ifft=scipy.fft.irfft, **kwds)


def dtft(x, real=False, n=None, n0=0):
    """ ??? """
    if n is None:
        n = len(x)
    return RegularAxis(n0, 1, len(x)).spectrum(x, real=real, n=n)


def idtft(X_spectrum, _N_FULL=None, real=False, n0=0):
    """ ??? """
    # See note in FreqRegularAxis.ispectrum as to why there is no size
    # n override argument.
    if real:
        assert _N_FULL is not None
        assert len(X_spectrum) == _N_FULL//2 + 1
        N = _N_FULL
    else:
        N = len(X_spectrum)
    axis_t = RegularAxis(n0, 1, N)
    axis_Omega = axis_t.spectrum_axis(N, real=real)
    return axis_Omega.ispectrum(X_spectrum)
