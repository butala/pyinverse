import numpy as np

from .axis import RegularAxis


class Angle:
    def __init__(self, /, *, rad=None, deg=None):
        """
        """
        if rad is None and deg is None:
            assert False
        if rad is None:
            self.deg = deg % 360.
            self.rad = np.radians(deg) % (2*np.pi)
        if deg is None:
            self.rad = rad % (2*np.pi)
            self.deg = np.degrees(rad) % 360.

    def __repr__(self):
        return f'<{self.__class__.__name__} deg={self.deg} rad={self.rad}>'

    def __str__(self):
        return f'{self.__class__.__name__}: degrees={self.deg} radians={self.rad}>'

    def __mul__(self, other):
        return Angle(rad=self.rad * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return Angle(rad=self.rad / other)

    def __rtruediv__(self, other):
        return self.__div__(other)

    def __add__(self, other):
        try:
            return Angle(deg=self.deg + other.deg)
        except:
            return Angle(rad=self.rad + other.rad)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        try:
            return Angle(deg=self.deg - other.deg)
        except:
            return Angle(rad=self.rad - other.rad)

    def __rsub__(self, other):
        return self.__sub__(other)



    @property
    def sin(self):
        try:
            return self._sin
        except AttributeError:
            self._sin = np.sin(self.rad)
            return self.sin

    @property
    def cos(self):
        try:
            return self._cos
        except AttributeError:
            self._cos = np.cos(self.rad)
            return self.cos

    @property
    def tan(self):
        try:
            return self._tan
        except AttributeError:
            self._tan = np.tan(self.rad)
            return self.tan


class AngleRegularAxis(RegularAxis):
    @property
    def rad(self):
        return RegularAxis(self.x0.rad, self.T.rad, self.N)

    @property
    def deg(self):
        return RegularAxis(self.x0.deg, self.T.deg, self.N)


if __name__ == '__main__':
    x = Angle(deg=45)
    print(x)
    print(x.sin)
