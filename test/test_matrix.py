'''
The generic Matrix data object of iteras, and SPECULA's Recmat derived from it.
'''
import specula
specula.init(0)  # Default target device

import os
import tempfile
import unittest

import numpy as np

from iteras import cpuArray
from iteras.calib_manager import CalibManager
from iteras.data_objects.matrix import Matrix
from iteras.lib import storage
from specula.data_objects.recmat import Recmat

from test.specula_testlib import cpu_and_gpu

FORMATS = ['fits', 'npz', 'npy', 'hdf5']


class TestMatrix(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = np.arange(6, dtype=np.float32).reshape(2, 3)

    def _path(self, name, data_format):
        return os.path.join(self.tmp.name, name + storage.EXTENSIONS[data_format])

    @cpu_and_gpu
    def test_values(self, target_device_idx, xp):
        m = Matrix(self.data, target_device_idx=target_device_idx)
        self.assertIsInstance(m.matrix, xp.ndarray)
        self.assertEqual((m.nrows, m.ncols), (2, 3))
        m.set_value(self.data * 2)
        np.testing.assert_array_equal(cpuArray(m.get_value()), self.data * 2)
        self.assertIs(m.array_for_display(), m.matrix)
        with self.assertRaises(AssertionError):
            m.set_value(np.zeros((3, 2)))

    @cpu_and_gpu
    def test_save_restore(self, target_device_idx, xp):
        m = Matrix(self.data, target_device_idx=target_device_idx)
        for data_format in FORMATS:
            with self.subTest(data_format=data_format):
                path = self._path(f'm{target_device_idx}', data_format)
                m.save(path)
                restored = Matrix.restore(path, target_device_idx=target_device_idx)
                self.assertIs(type(restored), Matrix)
                np.testing.assert_array_equal(cpuArray(restored.matrix), self.data)
                with storage.read_object(path) as f:
                    self.assertEqual(f.header['OBJ_TYPE'], 'Matrix')

    def test_missing_matrix(self):
        path = self._path('none', 'npz')
        storage.write_object(path, {'VERSION': 1}, {'other': self.data})
        with self.assertRaisesRegex(KeyError, 'matrix'):
            Matrix.restore(path)

    def test_wrong_version(self):
        path = self._path('v2', 'npz')
        storage.write_object(path, {'VERSION': 2}, {'matrix': self.data})
        with self.assertRaisesRegex(ValueError, 'version'):
            Matrix.restore(path)

    def test_calib_subdir(self):
        # 'matrix/' in iteras, 'rec/' in SPECULA, where matrices are reconstruction matrices
        cm = CalibManager('/root')
        self.assertEqual(cm.filename('Matrix', 'x'), os.path.join('/root', 'rec/', 'x.fits'))


class TestRecmatIsAMatrix(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = np.arange(6, dtype=np.float32).reshape(3, 2)

    def test_recmat_attribute_is_the_matrix(self):
        rec = Recmat(self.data, norm_factor=2.0, target_device_idx=-1)
        self.assertIsInstance(rec, Matrix)
        self.assertIs(rec.recmat, rec.matrix)
        self.assertEqual(rec.nmodes, 3)
        rec.recmat = np.zeros((2, 2), dtype=np.float32)
        self.assertEqual(rec.matrix.shape, (2, 2))
        # Not an array of its own: copied once by transferDataTo()
        self.assertNotIn('recmat', vars(rec))

    def test_recmat_files_read_as_matrix(self):
        '''A Recmat file is a Matrix file, with the reconstruction information'''
        rec = Recmat(self.data, norm_factor=2.0, target_device_idx=-1)
        for data_format in FORMATS:
            with self.subTest(data_format=data_format):
                path = os.path.join(self.tmp.name, 'rec' + storage.EXTENSIONS[data_format])
                rec.save(path)
                np.testing.assert_array_equal(cpuArray(Matrix.restore(path, target_device_idx=-1).matrix),
                                              self.data)
                restored = Recmat.restore(path, target_device_idx=-1)
                self.assertIs(type(restored), Recmat)
                self.assertEqual(restored.norm_factor, 2.0)
                with storage.read_object(path) as f:
                    self.assertEqual(f.header['OBJ_TYPE'], 'Recmat')

    def test_recmat_files_of_previous_versions(self):
        '''The matrix was named 'recmat', and FITS files had no extension names'''
        from astropy.io import fits
        for data_format in ('npz', 'npy', 'hdf5'):
            with self.subTest(data_format=data_format):
                path = os.path.join(self.tmp.name, 'old' + storage.EXTENSIONS[data_format])
                storage.write_object(path, {'VERSION': 1, 'NORMFACT': 0.5}, {'recmat': self.data})
                restored = Recmat.restore(path, target_device_idx=-1)
                np.testing.assert_array_equal(cpuArray(restored.recmat), self.data)
                self.assertEqual(restored.norm_factor, 0.5)
                # Also as a Matrix, as MVM and SoftLimiter restore them
                restored = Matrix.restore(path, target_device_idx=-1)
                np.testing.assert_array_equal(cpuArray(restored.matrix), self.data)

        path = os.path.join(self.tmp.name, 'old.fits')
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['NORMFACT'] = 0.5
        fits.writeto(path, np.zeros(2), hdr)
        fits.append(path, self.data)
        for klass in (Recmat, Matrix):
            with self.subTest(klass=klass.__name__):
                restored = klass.restore(path, target_device_idx=-1)
                np.testing.assert_array_equal(cpuArray(restored.matrix), self.data)


class TestIntmatIsAMatrix(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = np.arange(6, dtype=np.float32).reshape(3, 2)   # [slopes, modes]

    def test_intmat_attribute_is_the_matrix(self):
        from specula.data_objects.intmat import Intmat
        im = Intmat(self.data, target_device_idx=-1)
        self.assertIsInstance(im, Matrix)
        self.assertIs(im.intmat, im.matrix)
        self.assertEqual((im.nslopes, im.nmodes), (3, 2))
        im.set_nmodes(4)
        self.assertEqual(im.matrix.shape, (3, 4))
        self.assertNotIn('intmat', vars(im))
        # Empty interaction matrix of a given size
        self.assertEqual(Intmat(nmodes=2, nslopes=5, target_device_idx=-1).matrix.shape, (5, 2))

    def test_fits_layout_unchanged(self):
        from astropy.io import fits
        from specula.data_objects.intmat import Intmat
        path = os.path.join(self.tmp.name, 'im.fits')
        Intmat(self.data, pupdata_tag='pup', target_device_idx=-1).save(path)
        with fits.open(path) as hdul:
            self.assertIsNone(hdul[0].data)
            np.testing.assert_array_equal(hdul[1].data, self.data)
            self.assertEqual(hdul[0].header['PUP_TAG'], 'pup')

    def test_intmat_files_read_as_matrix(self):
        from specula.data_objects.intmat import Intmat
        im = Intmat(self.data, pupdata_tag='pup', target_device_idx=-1)
        for data_format in FORMATS:
            with self.subTest(data_format=data_format):
                path = os.path.join(self.tmp.name, 'im' + storage.EXTENSIONS[data_format])
                im.save(path)
                np.testing.assert_array_equal(cpuArray(Matrix.restore(path, target_device_idx=-1).matrix),
                                              self.data)
                restored = Intmat.restore(path, target_device_idx=-1)
                self.assertIs(type(restored), Intmat)
                self.assertEqual(restored.pupdata_tag, 'pup')

    def test_intmat_files_of_previous_versions(self):
        '''The matrix was named 'intmat' in the npz, npy and HDF5 files'''
        from specula.data_objects.intmat import Intmat
        for data_format in ('npz', 'npy', 'hdf5'):
            with self.subTest(data_format=data_format):
                path = os.path.join(self.tmp.name, 'old' + storage.EXTENSIONS[data_format])
                storage.write_object(path, {'VERSION': 1}, {'intmat': self.data})
                for klass in (Intmat, Matrix):
                    np.testing.assert_array_equal(
                        cpuArray(klass.restore(path, target_device_idx=-1).matrix), self.data)

    def test_soft_limiter_takes_an_intmat(self):
        '''The reprojector of SoftLimiter is an interaction matrix'''
        from specula.data_objects.intmat import Intmat
        from iteras.processing_objects.soft_limiter import SoftLimiter
        estimator = Matrix(self.data.T, target_device_idx=-1)               # (modes, slopes)
        limiter = SoftLimiter(matrix_list=[estimator, Intmat(self.data, target_device_idx=-1)],
                              target_device_idx=-1)
        np.testing.assert_array_equal(cpuArray(limiter.reprojector), self.data)


if __name__ == '__main__':
    unittest.main()
