import numpy as np

from .axis import RegularAxis


class Frequency:
    def __init__(self, /, *, f=None, w=None):
        """
        f: Hz
        w: rad/s (omega)
        """
        if f is None and w is None:
            assert False
        if w is None:
            self.f = f
            self.w = 2 * np.pi * f
        if f is None:
            self.f = w / (2 * np.pi)
            self.w = w

    def __repr__(self):
        return f'<{self.__class__.__name__} f={self.f} w={self.w}>'

    def __str__(self):
        return f'{self.__class__.__name__}: omega={self.w} rad/s f={self.f} Hz>'

    def __mul__(self, other):
        return Frequency(f=self.f * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return Frequency(f=self.f / other)

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __add__(self, other):
        try:
            return Frequency(f=self.f + other.f)
        except:
            return Frequency(w=self.w + other.w)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        try:
            return Frequency(f=self.f - other.f)
        except:
            return Frequency(w=self.w - other.w)

    def __rsub__(self, other):
        return self.__sub__(other)


class FrequencyRegularAxis(RegularAxis):
    @property
    def f(self):
        return RegularAxis(self.x0.f, self.T.f, self.N)

    @property
    def w(self):
        return RegularAxis(self.x0.w, self.T.w, self.N)


if __name__ == '__main__':
    x0 = Frequency(f=100)
    T = Frequency(f=20)
    N = 15

    x = FrequencyRegularAxis(x0, T, N)

    print(x)
    print(x.f)
    print(x.w)
