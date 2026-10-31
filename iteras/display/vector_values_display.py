import numpy as np

from iteras.display.base_display import BaseDisplay
from iteras.connections import InputValue
from iteras.base_value import BaseValue
from iteras import cpuArray


class VectorValuesDisplay(BaseDisplay):
    """
    Display of the current values of a vector, as a function of their index.

    Parameters:
    -----------
    xrange : tuple, optional
        Fixed X range (xmin, xmax). Default: the indices of the vector
    yrange : tuple, optional
        Fixed Y range (ymin, ymax). Use (0, 0) or None to scale to the data
    xlabel, ylabel : str
        Labels of the axes
    """

    # Name of the input
    input_name = 'value'

    def __init__(self,
                 title='Vector Values',
                 figsize=(6, 3),
                 xrange=None,
                 yrange=(0, 0),
                 xlabel='Index',
                 ylabel='Value',
                 window: int=None,
                 subplot: int=111,
                 window_xy=None,
                 ):

        super().__init__(
            title=title,
            figsize=figsize,
            window=window,
            subplot=subplot,
            window_xy=window_xy,
        )

        self._xrange = xrange
        self._yrange = yrange
        self._xlabel = xlabel
        self._ylabel = ylabel
        self.line = None

        # Setup input
        self.input_key = self.input_name  # Used by base class
        self.inputs[self.input_name] = InputValue(type=BaseValue)

    def _fixed_yrange(self):
        return self._yrange is not None and np.sum(np.abs(self._yrange)) > 0

    def _update_display(self, vector):
        y = cpuArray(vector.value)
        x = np.arange(len(y))

        if self.line is None:
            # First time: create line
            self.line = self.ax.plot(x, y, '.-')[0]

            # Set fixed X range if specified
            if self._xrange is not None:
                self.ax.set_xlim(self._xrange[0], self._xrange[1])
            elif len(x) > 0:
                self.ax.set_xlim(0, len(x) - 1)

            # Set fixed Y range if specified
            if self._fixed_yrange():
                self.ax.set_ylim(self._yrange[0], self._yrange[1])
            else:
                # Auto-scale based on data
                self.ax.set_ylim(y.min() * 1.1, y.max() * 1.1)

            # Add reference line at y=0
            self.ax.axhline(y=0, color='grey', linestyle='--',
                          dashes=(4, 8), linewidth=0.5, alpha=0.7)

            # Set labels
            self.ax.set_xlabel(self._xlabel)
            self.ax.set_ylabel(self._ylabel)
        else:
            # Update existing line
            self.line.set_xdata(x)
            self.line.set_ydata(y)

            # Update X limits if vector size changed or xrange is set
            if self._xrange is not None:
                self.ax.set_xlim(self._xrange[0], self._xrange[1])
            elif len(x) > 0:
                self.ax.set_xlim(0, len(x) - 1)

            # Update Y limits if auto-scaling
            if not self._fixed_yrange():
                self.ax.set_ylim(y.min() * 1.1, y.max() * 1.1)
