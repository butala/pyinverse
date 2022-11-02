from .axis import RegularAxis


class RegularAxes:
    def __init__(self, *axis_list):
        self.axis_list = axis_list

    @classmethod
    def linspace(cls, *linspace_list):
        """ ??? """
        return cls(*[RegularAxis.linspace(*x) for x in linspace_list])

    def __repr__(self):
        return f'<{self.__class__.__name__} ' + ' '.join([f'<axis_{i}>: {repr(axis_i)}' for i, axis_i in enumerate(self.axis_list, 1)]) + '>'

    def __str__(self):
        return f'{self.__class__.__name__}:\n' + '\n'.join([f'axis {i}: {str(axis_i)}' for i, axis_i in enumerate(self.axis_list, 1)])


class RegularAxes3(RegularAxes):
    """Regular, i.e., equally spaced, points on a grid.

    Args:
        axis_x (RegularAxis): horizontal axis
        axis_y (RegularAxis): vertical axis
        axis_z (RegularAxis): height axis

    """
    def __init__(self, axis_x, axis_y, axis_z):
        """ ??? """
        super().__init__(axis_x, axis_y, axis_z)


    @property
    def axis_x(self):
        return self.axis_list[0]

    @property
    def axis_y(self):
        return self.axis_list[1]

    @property
    def axis_z(self):
        return self.axis_list[2]

    @classmethod
    def linspace(cls, linspace1, linspace2, linspace3):
        """ ??? """
        return super().linspace(linspace1, linspace2, linspace3)

    def __repr__(self):
        return f'<{self.__class__.__name__} <axis_x: {repr(self.axis_x)}> <axis_y: {repr(self.axis_y)}> <axis_z {repr(self.axis_z)}>>'

    def __str__(self):
        return f'{self.__class__.__name__}:\naxis x: {str(self.axis_x)}\naxis y: {str(self.axis_y)}\naxis z: {str(self.axis_z)}'
