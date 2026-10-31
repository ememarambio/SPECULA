
import iteras
from iteras import np
from astropy.io import fits
from iteras.base_data_obj import BaseDataObj
from iteras.lib import storage


class BaseValue(BaseDataObj):
    def __init__(self, description='', value=None, target_device_idx=None, precision=None):
        """
        Initialize the base value object.

        Parameters:
        description (str, optional)
        value (any, optional): data to store. If not set, the value is initialized to None.
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.description = description
        if value is not None:
            # if it is a scalar, convert to the appropriate scalar type
            if np.isscalar(value):
                self.value = self.dtype(value)
            else:
                self.value = self.to_xp(value, force_copy=True, dtype=self.dtype)
        else:
            self.value = None

    def get_value(self):
        return self.value

    def set_value(self, val):
        if self.value is not None:
            if np.isscalar(self.value):
                self.value = self.dtype(val)
            else:
                self.value[...] = self.to_xp(val)
        else:
            self.value = self.to_xp(val, force_copy=True, dtype=self.dtype)

    def save(self, filename, overwrite=False):
        hdr = self.get_fits_header()
        if type(self.value) in iteras.array_types:
            hdr['NDARRAY'] = 1
            storage.write_object(filename, hdr, {'value': self.value}, layout='primary',
                                 overwrite=overwrite)
        else:
            hdr['NDARRAY'] = 0
            if self.value is not None:
                hdr['VALUE'] = str(self.value)  # Store as string for simplicity
            storage.write_object(filename, hdr, {}, layout='placeholder', overwrite=overwrite)

    @staticmethod
    def restore(filename, target_device_idx=None):
        v = BaseValue(target_device_idx=target_device_idx)
        with storage.read_object(filename) as f:
            hdr = f.header
            if hdr['NDARRAY']:
                data = f.read('value', 0)
                # Float arrays follow the object precision; other types and the
                # (host) location are left unchanged
                if np.issubdtype(data.dtype, np.floating):
                    v.value = data.astype(v.dtype)
                else:
                    v.value = data.copy()
            else:
                value_str = hdr.get('VALUE', None)
                if value_str is not None:
                    v.value = eval(value_str)  # Convert back from string to original type
        return v

    def array_for_display(self):
        return self.value

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['OBJ_TYPE'] = 'BaseValue'
        return hdr
