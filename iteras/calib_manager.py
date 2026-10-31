
import os

from iteras import registry
from iteras.lib import storage
from iteras.log import get_iteras_logger


class CalibManager():
    def __init__(self, root_dir):
        """
        Initialize the calibration manager object.

        Parameters:
        root_dir (str): Root path of the calibration tree

        Files are stored in a subdirectory of root_dir depending on their
        data type (or on the class that computes them). Plugins register
        their subdirectories with iteras.registry.register_calib_subdirs().
        """
        super().__init__()
        # Subdirectories of the data types, extended by plugins (see iteras.registry)
        self._subdirs = registry.calib_subdirs()
        self.root_dir = root_dir
        self.logger = get_iteras_logger('iteras.calib_manager')

    def root_subdir(self, type):
        """
        Returns the full path to the subdirectory corresponding to the given type.

        Parameters:
            type (str): The key representing the calibration data type.

        Returns:
            str: The absolute path to the subdirectory for the specified type.
        """
        return os.path.join(self.root_dir, self._subdirs[type])

    def filename(self, subdir, name):
        """
        Construct the full file path for a calibration file.

        Parameters:
            subdir (str): The key representing the calibration data type or subdirectory.
            name (str): The base name of the file, usually without extension.

        Returns:
            str: The path of the file. If *name* has no FITS, npz or HDF5 extension,
            the existing file with that name and one of these extensions (see
            iteras.lib.storage.find_object_file()), or, if there is none, the
            name with the extension of the default format ('.fits' unless
            ITERAS_DATA_FORMAT says otherwise).
        """
        fname = os.path.join(self.root_dir, self._subdirs[subdir], name)
        if storage.object_format(fname) is None:
            fname = storage.find_object_file(fname) or storage.object_filename(fname)
        return fname

    def writefits(self, subdir, name, data):
        """
        Write data to a file, in FITS format unless the file name has the
        extension of another format (see filename()).

        Parameters:
            subdir (str): The key representing the calibration data type or subdirectory.
            name (str): The base name of the file (without extension).
            data (array-like): The data to be written to the file.
        """
        filename = self.filename(subdir, name)
        # In FITS files, the data is in the primary HDU
        storage.write_object(filename, {}, {'data': data}, layout='primary', overwrite=True)

    def readfits(self, subdir, name):
        """
        Read data from a file in any of the formats of filename().

        Parameters:
            subdir (str): The key representing the calibration data type or subdirectory.
            name (str): The base name of the file (without extension).

        Returns:
            array-like: The data read from the file.
        """
        filename = self.filename(subdir, name)
        self.logger.info(f'Reading: {filename}')
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        with storage.read_object(filename) as f:
            return f.read('data', 0)

    def write_data(self, name, data):
        """
        Write data to a file in the 'data' subdirectory (see writefits()).

        Parameters:
            name (str): The base name of the file (without extension).
            data (array-like): The data to be written to the file.
        """
        self.writefits('data', name, data)

    def read_data(self, name):
        """
        Read data from a file in the 'data' subdirectory (see readfits()).

        Parameters:
            name (str): The base name of the file (without extension).

        Returns:
            array-like: The data read from the file.
        """
        return self.readfits('data', name)

    def __repr__(self):
        return 'Calibration manager'

