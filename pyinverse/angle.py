import numpy as np


class Angle:
    def __init__(self, /, *, rad=None, deg=None):
        """
        """
        if rad is None and deg is None:
            assert False
        if rad is None:
            self.deg = deg
            self.rad = np.radians(deg)
        if deg is None:
            self.rad = rad
            self.deg = np.degrees(rad)

    def __repr__(self):
        return f'<{self.__class__.__name__} deg={self.deg} rad={self.rad}>'

    def __str__(self):
        return f'{self.__class__.__name__}: degrees={self.deg} radians={self.rad}>'

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


if __name__ == '__main__':
    x = Angle(deg=45)
    print(x)
    print(x.sin)
