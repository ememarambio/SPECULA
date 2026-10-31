from astropy.io import fits

from iteras.base_data_obj import BaseDataObj
from iteras.lib import storage


# Names of the matrix in the npz, npy and HDF5 files: SPECULA saved its
# reconstruction and interaction matrices (Recmat, Intmat) as 'recmat' and
# 'intmat' before they derived from Matrix
_ARRAY_NAMES = ('matrix', 'recmat', 'intmat')


def matrix_array_name(reader):
    '''Name of the matrix in a file opened with storage.read_object()'''
    for name in _ARRAY_NAMES:
        if reader.has(name, 1):
            return name
    return _ARRAY_NAMES[0]   # let load() report the missing array


class Matrix(BaseDataObj):
    """
    Matrix data object.
    This class holds a 2D matrix, for example the matrix of a linear
    operator applied by :class:`~iteras.processing_objects.mvm.MVM`.
    Plugins can derive from it to add domain-specific information.
    """
    def __init__(self,
                 matrix,
                 target_device_idx: int=None,
                 precision: int=None):
        """
        Initialize a :class:`~iteras.data_objects.matrix.Matrix` object.

        Parameters
        ----------
        matrix : array_like
            The 2D matrix, of shape (rows, columns).
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.matrix = self.to_xp(matrix, dtype=self.dtype)

    def get_value(self):
        '''
        Get the matrix as a numpy/cupy array
        '''
        return self.matrix

    def set_value(self, v):
        '''
        Set new values for the matrix
        Arrays are not reallocated
        '''
        assert v.shape == self.matrix.shape, \
            f"Error: input array shape {v.shape} does not match matrix shape {self.matrix.shape}"
        self.matrix[:] = self.to_xp(v)

    @property
    def nrows(self):
        return self.matrix.shape[0]

    @property
    def ncols(self):
        return self.matrix.shape[1]

    def array_for_display(self):
        return self.matrix

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['OBJ_TYPE'] = type(self).__name__
        return hdr

    def _arrays_to_save(self):
        '''Arrays written by save(), by name: derived classes can add their own'''
        return {'matrix': self.matrix}

    def save(self, filename, overwrite=False):
        filename = storage.object_filename(filename)
        storage.write_object(filename, self.get_fits_header(), self._arrays_to_save(),
                             layout='placeholder', overwrite=overwrite)

    @staticmethod
    def from_header(hdr, target_device_idx=None):
        raise NotImplementedError

    @staticmethod
    def restore(filename, target_device_idx=None):
        '''
        Restore a matrix from a file written by save(), also by the save()
        of a derived class: the derived information is not restored.
        '''
        with storage.read_object(filename) as f:
            version = int(f.header['VERSION'])
            if version != 1:
                raise ValueError(f"Error: unknown version {version} in file {filename}")
            matrix = f.load(matrix_array_name(f), 1, target_device_idx=target_device_idx)
        return Matrix(matrix, target_device_idx=target_device_idx)
