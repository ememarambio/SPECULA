
from astropy.io import fits

from iteras.base_data_obj import BaseDataObj
from iteras.lib import storage


class TimeHistory(BaseDataObj):
    """
    Time history data object.
    This class holds the time history of a variable, such as a parameter changing during the simulation.
    The time history is stored as a 1D array,
    """
    def __init__(self,
                 time_history,
                 target_device_idx: int=None,
                 precision:int =None):
        """
        Initialize a :class:`~iteras.data_objects.time_history.TimeHistory` object.
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.time_history = self.to_xp(time_history)

    def get_value(self):
        return self.time_history

    def set_value(self, val):
        self.time_history[...] = self.to_xp(val)

    def save(self, filename):
        """Saves the :class:`~iteras.data_objects.time_history.TimeHistory` data to a file."""
        storage.write_object(filename, self.get_fits_header(), {'time_history': self.time_history},
                             layout='primary', overwrite=False)

    @staticmethod
    def restore(filename, target_device_idx=None):
        """Restores the :class:`~iteras.data_objects.time_history.TimeHistory` data from a file."""
        with storage.read_object(filename) as f:
            version = f.header.get('VERSION')
            if version != 1:
                raise ValueError(f"Unknown version {version} in file {filename}")
            data = f.read('time_history', 0)
        return TimeHistory(data, target_device_idx=target_device_idx)

    def array_for_display(self):
        return self.time_history
    
    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['OBJ_TYPE'] = 'TimeHistory'
        return hdr
    
    @staticmethod
    def from_header(hdr, target_device_idx=None):
        raise NotImplementedError
