from iteras.display.vector_values_display import VectorValuesDisplay


class ModesDisplay(VectorValuesDisplay):
    """
    Display of modal coefficients, in nm, as a function of the mode index:
    a :class:`~iteras.display.vector_values_display.VectorValuesDisplay`
    with the input ``modes``.
    """

    input_name = 'modes'

    def __init__(self,
                 title='Modes Display',
                 figsize=(6, 3),
                 xrange=None,
                 yrange=(-500, 500),
                 window: int=None,
                 subplot: int=111,
                 window_xy=None,
                 ):
        super().__init__(title=title, figsize=figsize, xrange=xrange, yrange=yrange,
                         xlabel='Mode index', ylabel='Mode value (nm)',
                         window=window, subplot=subplot, window_xy=window_xy)
