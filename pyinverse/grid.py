import imageio
from tqdm import tqdm
import numpy as np
import scipy.fft


from .axis import Order, RegularAxis, FFTRegularAxis, RFFTRegularAxis
from .rect import srect_2D_proj


class RegularGrid:
    """Regular, i.e., equally spaced, points on a grid.

    Args:
        axis_x (RegularAxis): horizontal axis
        axis_y (RegularAxis): vertical axis

    """
    def __init__(self, axis_x, axis_y):
        """ """
        self.axis_x = axis_x
        self.axis_y = axis_y

    @classmethod
    def image(cls, x):
        """ ??? """
        Ny, Nx = x.shape
        return cls(RegularAxis(0, 1, Nx), RegularAxis(0, 1, Ny))

    @classmethod
    def from_image(cls, im_fname):
        """ ??? """
        x = imageio.imread(im_fname)
        assert x.ndim == 2
        return cls.image(x), x

    @classmethod
    def linspace(cls, linspace1, linspace2):
        """ ??? """
        return cls(RegularAxis.linspace(*linspace1),
                   RegularAxis.linspace(*linspace2))

    def __repr__(self):
        return f'<{self.__class__.__name__} <axis_x: {repr(self.axis_x)}> <axis_y: {repr(self.axis_y)}>>'

    def __str__(self):
        return f'{self.__class__.__name__}:\naxis x: {str(self.axis_x)}\naxis y: {str(self.axis_y)}'

    @property
    def centers(self):
        """ ??? """
        try:
            return self._centers
        except AttributeError:
            # For reference on all the numpy grid functions: https://stackoverflow.com/questions/12402045/mesh-grid-functions-in-python-meshgrid-mgrid-ogrid-ndgrid
            self._centers = np.meshgrid(self.axis_x.centers, self.axis_y.centers)
            return self.centers

    @property
    def borders(self):
        """ ??? """
        try:
            return self._borders
        except AttributeError:
            if self.axis_x._order == Order.INCREASING and self.axis_y._order == Order.INCREASING:
                self._borders = np.meshgrid(self.axis_x.borders, self.axis_y.borders)
                return self.borders
            else:
                # What to do if an axis is in decreasing order?
                assert False

    @property
    def shape(self):
        """Return the tuple with the number of vertical and horizontal sample points."""
        return self.axis_y.N, self.axis_x.N

    @property
    def N_fast(self):
        """ ??? """
        # This also works for rfft?
        return self.axis_x.N_fast, self.axis_y.N_fast

    def __iter__(self):
        """
        Iterate over each grid center coordinate in proper order,
        i.e., the associated (i,j)th matrix element is located at
        coordinate (axis_y[i], axis_x[j]).
        """
        for i in self.axis_y:
            for j in self.axis_x:
                yield (i, j)

    def __getitem__(self, index):
        """Return the (ith, jth) sample point."""
        i, j = index
        return self.axis_y[i], self.axis_x[j]

    def scale(self, Sx, Sy):
        """ ??? """
        return RegularGrid(self.axis_x.scale(Sx), self.axis_y.scale(Sy))

    def Hz(self, axis=None):
        """ ??? """
        # Need to have units to know if this is valid
        scale_map = {None: (1 / (2*np.pi), 1 / (2*np.pi)),
                     0:    (1, 1 / (2*np.pi)),
                     1:    (1 / (2*np.pi), 1)}
        return self.scale(*scale_map[axis])

    # (axis_x order, axis_y, order) -> ((increasing?, increasing?), axes)
    _XFORM_MAP = {(Order.FFT, Order.FFT):        ((True, True),  None),
                  (Order.FFT, Order.INCREASING): ((True, False), 1),
                  (Order.INCREASING, Order.FFT): ((False, True), 0),
                  (Order.INCREASING, Order.INCREASING): ((False, False), False)}

    # (axis_x order, axis_y, order) -> appropgriate fftshift
    _SHIFT_MAP = {(Order.FFT, Order.FFT):        lambda x: scipy.fft.fftshift(x),
                  (Order.FFT, Order.INCREASING): lambda x: scipy.fft.fftshift(x, axes=1),
                  (Order.INCREASING, Order.FFT): lambda x: scipy.fft.fftshift(x, axes=0),
                  (Order.INCREASING, Order.INCREASING): lambda x: x}

    def increasing(self, x=None):
        """ ??? """
        if x is not None:
            assert x.shape == self.shape

        (increasing_x, increasing_y), axes = self._XFORM_MAP[(self.axis_x._order,
                                                              self.axis_y._order)]

        grid = RegularGrid(self.axis_x.increasing() if increasing_x else self.axis_x,
                           self.axis_y.increasing() if increasing_y else self.axis_y)

        if x is not None:
            x = self._SHIFT_MAP[(self.axis_x._order,
                                 self.axis_y._order)](x)
            return grid, x
        else:
            return grid


    def plot(self, ax, X, interpolation='none', flip_y=False, **kwds):
        """Display the 2-D array *X* as an image on axis *ax* taking into
        account the grid orientation. Additional arguments *kwds* are
        passed to :func:`imshow`.

        For details: https://matplotlib.org/3.3.1/tutorials/intermediate/imshow_extent.html

        """
        if self.axis_x._order != Order.INCREASING or self.axis_y._order != Order.INCREASING:
            grid_increasing, X_increasing = self.increasing(X)
            return grid_increasing.plot(ax, X_increasing, interpolation=interpolation, **kwds)

        assert self.shape == X.shape
        assert 'origin' not in kwds
        assert 'extent' not in kwds
        x_extent = [self.axis_x.borders[0], self.axis_x.borders[-1]]
        if flip_y:
            kwds['origin'] = 'upper'
            y_extent = [self.axis_y.borders[-1], self.axis_y.borders[0]]
        else:
            kwds['origin'] = 'lower'
            y_extent = [self.axis_y.borders[0], self.axis_y.borders[-1]]
        kwds['extent'] = x_extent + y_extent
        return ax.imshow(X, interpolation=interpolation, **kwds)

    def imshow(self, ax, X, interpolation='none', **kwds):
        """ ??? """
        return self.plot(ax, X, interpolation=interpolation, flip_y=True, **kwds)


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
        if real:
            return RealFreqRegularGrid(grid_freq_axis[1], grid_freq_axis[0], self)
        else:
            return FreqRegularGrid(grid_freq_axis[1], grid_freq_axis[0], self)


    # (real, axis) -> appropriate (r)fft function
    _FFT_MAP = {(True, None):  lambda x, s: scipy.fft.rfft2(x, s=s),
                (True, 0):     lambda x, s: scipy.fft.rfft(x, n=s[0], axis=0),
                (True, 1):     lambda x, s: scipy.fft.rfft(x, n=s[1], axis=1),
                (False, None): lambda x, s: scipy.fft.fft2(x, s=s),
                (False, 0):    lambda x, s: scipy.fft.fft(x, n=s[0], axis=0),
                (False, 1):    lambda x, s: scipy.fft.fft(x, n=s[1], axis=1)}


    def spectrum(self, x, s=None, axis=None, real=False):
        """ ??? """
        assert x.shape == self.shape
        assert self.axis_x._order == Order.INCREASING and self.axis_y._order == Order.INCREASING
        if s is None:
            s = self.shape
        elif s < self.shape:
            raise NotImplementedError()
        X_spectrum = self._FFT_MAP[(real, axis)](x, s)
        grid_freq = self.spectrum_grid(s=s, axis=axis, real=real)
        if axis is None:
            P = np.exp(-1j*(grid_freq.centers[0]*self.axis_x.x0 + grid_freq.centers[1]*self.axis_y.x0))
            X_spectrum *= P * self.axis_x.T * self.axis_y.T
        elif axis == 0:
            p = np.exp(-1j*grid_freq.axis_y.centers*self.axis_y.x0)
            X_spectrum *= np.atleast_2d(p).T * self.axis_y.T
        elif axis == 1:
            p = np.exp(-1j*grid_freq.axis_x.centers*self.axis_x.x0)
            X_spectrum *= np.atleast_2d(p) * self.axis_x.T
        else:
            assert False
        return grid_freq, X_spectrum


    def sinogram(self, sinogram_grid, A, deg=True):
        """Calculate the sinogram of the array *A* which contains the
        densities in each of the grid elements. The projections are
        calculated according to *sinogram_grid* which specifies the
        projection angles (the x-axis) and projection sample points
        (the y-axis). The angular units of the projections are in
        given in degrees if `deg=True`. Return an array (shape = # of
        projection samples x # of angles).

        """
        assert self.shape == A.shape
        if deg:
            theta_rad = np.radians(sinogram_grid.axis_x.centers)
        else:
            theta_rad = sinogram_grid.axis_x.centers
        proj = np.zeros(sinogram_grid.shape)
        T_x = self.axis_x.T
        T_y = self.axis_y.T
        for k, theta_k in tqdm(enumerate(theta_rad), total=len(theta_rad)):
            cos_theta = np.cos(theta_k)
            sin_theta = np.sin(theta_k)
            for j, a in enumerate(self.axis_x.centers):
                for i, b in enumerate(self.axis_y.centers):
                    proj[:, k] += srect_2D_proj([theta_k], sinogram_grid.axis_y.centers - a*cos_theta - b*sin_theta, 1/T_x, 1/T_y).flat * A[i, j]
        return proj


# grid_check is abstract
# can you make _ifft an abstract property?
class FreqRegularGridBase(RegularGrid):
    def __init__(self, axis_x, axis_y, grid_s):
        super().__init__(axis_x, axis_y)
        self.grid_s = grid_s

    def grid_check(self, axis):
        type_x, type_y = self._AXIS_MAP[axis]
        assert isinstance(self.axis_x, type_x) and isinstance(self.axis_y, type_y)

    def ispectrum(self, X_spectrum, axis=None):
        """ ??? """
        assert X_spectrum.shape == self.shape
        self.grid_check(axis)

        if axis is None:
            s = (self.axis_y._N_FULL, self.axis_x._N_FULL)
            P = np.exp(1j*(self.centers[0]*self.axis_x.axis_t.x0 + self.centers[1]*self.axis_y.axis_t.x0))
            x = self._IFFT2(X_spectrum * P / (self.grid_s.axis_x.T * self.grid_s.axis_y.T), s=s)
        elif axis == 0:
            n = self.axis_y._N_FULL
            p = np.exp(1j*self.axis_y.centers*self.axis_y.axis_t.x0)
            x = self._IFFT(X_spectrum * np.atleast_2d(p).T / self.grid_s.axis_y.T, n=n, axis=0)
        elif axis == 1:
            n = self.axis_x._N_FULL
            p = np.exp(1j*self.axis_x.centers*self.axis_x.axis_t.x0)
            x = self._IFFT(X_spectrum * np.atleast_2d(p) / self.grid_s.axis_x.T, n=n, axis=1)
        else:
            assert False

        x = x[:self.grid_s.axis_y.N, :self.grid_s.axis_x.N]
        return self.grid_s, x


class FreqRegularGrid(FreqRegularGridBase):
    @property
    def _IFFT2(self):
        return scipy.fft.ifft2

    @property
    def _IFFT(self):
        return scipy.fft.ifft

    _AXIS_MAP = {None: (FFTRegularAxis, FFTRegularAxis),
                 0:    (RegularAxis, FFTRegularAxis),
                 1:    (FFTRegularAxis, RegularAxis)}


class RealFreqRegularGrid(FreqRegularGridBase):
    @property
    def _IFFT2(self):
        return scipy.fft.irfft2

    @property
    def _IFFT(self):
        return scipy.fft.irfft

    _AXIS_MAP = {None: (RFFTRegularAxis, FFTRegularAxis),
                 0:    (RegularAxis, RFFTRegularAxis),
                 1:    (RFFTRegularAxis, RegularAxis)}


def dtft2(x, axis=None, real=False, s=None, n0=(0, 0)):
    """ ??? """
    if s is None:
        s = x.shape
    axis_x = RegularAxis(n0[1], 1, x.shape[1])
    axis_y = RegularAxis(n0[0], 1, x.shape[0])
    grid = RegularGrid(axis_x, axis_y)
    return grid.spectrum(x, axis=axis, real=real, s=s)


def idtft2(x_dtft2, axis=None, _Nx_FULL=None, real=False, n0=(0, 0)):
    """ ??? """
    if real:
        Ny = x_dtft2.shape[0]
        assert x_dtft2.shape[1] == _Nx_FULL//2 + 1
        Nx = _Nx_FULL
    else:
        Ny, Nx = x_dtft2.shape
    axis_x = RegularAxis(n0[1], 1, Nx)
    axis_y = RegularAxis(n0[0], 1, Ny)
    grid = RegularGrid(axis_x, axis_y)
    grid_spectrum = grid.spectrum_grid(real=real)
    return grid_spectrum.ispectrum(x_dtft2, axis=axis)
