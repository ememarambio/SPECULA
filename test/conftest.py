import os

# Limit BLAS/OpenMP threads before numpy is imported by any test module.
# Test matrices are small, and with the default (one thread per core) the
# LAPACK calls are dominated by threading overhead, especially on a loaded machine.
for _var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '4')


import pytest

# Values set by init() (iteras.init(), also available as specula.init()).
# The framework reads them when objects are created, so each test must see
# the values set by its own file (see _restore_init_state())
_INIT_STATE = ['xp', 'cp', 'gpuEnabled', 'global_precision', 'float_dtype_list',
               'complex_dtype_list', 'gpu_float_dtype_list', 'gpu_complex_dtype_list',
               'array_types', 'float_dtype', 'complex_dtype', 'default_target_device_idx',
               'default_target_device', 'process_comm', 'process_rank']


# Values set by init() right after each test file was imported, by module node id
_module_init_state = {}


def _iteras_state():
    import iteras
    return {name: getattr(iteras, name) for name in _INIT_STATE}


def pytest_collectreport(report):
    '''
    Called after each test file is collected, that is, right after it has been
    imported: record the values set by the init() call at its top, if any.
    '''
    if report.nodeid.endswith('.py') and report.passed:
        _module_init_state[report.nodeid] = _iteras_state()


@pytest.fixture(autouse=True)
def _restore_init_state(request):
    '''
    Run each test with the values set by the init() call at the top of its
    file, and restore the previous values after the test.

    All the test files are imported before running any test, so without this
    the tests would see the values set by the last file imported (for example
    single precision instead of double), and a test calling init(), directly
    or through main_simul() (which uses single precision by default), would
    change the device and precision of the objects created by the following tests.
    '''
    import iteras
    saved = _iteras_state()
    module_state = _module_init_state.get(request.node.nodeid.split('::')[0])
    if module_state is not None:
        for name, value in module_state.items():
            setattr(iteras, name, value)
    yield
    for name, value in saved.items():
        setattr(iteras, name, value)



def _assert_same_restored(a, b, path='object'):
    '''Compare two restored data objects (or values) field by field'''
    import numpy as np
    from iteras import cpuArray
    from iteras.base_time_obj import BaseTimeObj
    if isinstance(a, BaseTimeObj) or hasattr(a, '__dict__') and not callable(a):
        assert type(a) is type(b), f'{path}: {type(a)} != {type(b)}'
        # Not data: device handles and GPU memory tracking (see BaseTimeObj)
        skip = {'logger', 'xp', '_target_device', 'PerformanceWarning', '_mem_before',
                '_mem_count_depth', '_mem_owner', '_mem_parent', 'gpu_bytes_used',
                'gpu_bytes_children'}
        for key in set(vars(a)) | set(vars(b)):
            if key not in skip:
                _assert_same_restored(getattr(a, key, None), getattr(b, key, None), f'{path}.{key}')
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), f'{path}: {a!r} != {b!r}'
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_same_restored(x, y, f'{path}[{i}]')
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), f'{path}: {a!r} != {b!r}'
        for key in a:
            _assert_same_restored(a[key], b[key], f'{path}[{key!r}]')
    elif hasattr(a, 'shape') and hasattr(a, 'dtype'):
        a, b = cpuArray(a), cpuArray(b)
        assert a.dtype == b.dtype, f'{path}: dtype {a.dtype} != {b.dtype}'
        np.testing.assert_array_equal(a, b, err_msg=path)
    else:
        assert a == b or (a != a and b != b), f'{path}: {a!r} != {b!r}'


def _checking_save(func):
    '''
    Wrapper of the save() method of a data object class, for
    ITERAS_TEST_CHECK_OBJECT_FORMATS=1: after saving an object to a FITS file,
    also save it to npz, npy and HDF5 files, restore the four files with the
    restore() method of its class, and check that the results are equal.
    '''
    import functools
    import tempfile
    from iteras.lib import storage

    @functools.wraps(func)
    def save(self, filename, *args, **kwargs):
        result = func(self, filename, *args, **kwargs)
        restore = getattr(type(self), 'restore', None)
        if storage.object_format(filename) not in ('fits', None) or restore is None \
                or not os.path.isfile(storage.object_filename(filename)):
            return result
        try:
            reference = restore(storage.object_filename(filename))
        except Exception:
            return result       # not restorable by itself (e.g. needs extra arguments)
        with tempfile.TemporaryDirectory() as tmpdir:
            for data_format in ('npz', 'npy', 'hdf5'):
                other = os.path.join(tmpdir, 'object' + storage.EXTENSIONS[data_format])
                func(self, other, *args, **kwargs)
                _assert_same_restored(reference, restore(other),
                                      f'{type(self).__name__} restored from {data_format}')
                _checked_classes.add((type(self).__name__, data_format))
        return result
    return save


# Classes and formats checked in ITERAS_TEST_CHECK_OBJECT_FORMATS mode
_checked_classes = set()


def pytest_configure(config):
    '''
    With ITERAS_TEST_CHECK_OBJECT_FORMATS=1, each data object saved to a FITS
    file by a test is also saved to npz, npy and HDF5 files, and the objects
    restored from the four files are compared (see test_object_file_formats.py).
    The save() methods of the data object classes defined from now on are wrapped.
    '''
    if os.environ.get('ITERAS_TEST_CHECK_OBJECT_FORMATS'):
        from iteras.base_data_obj import BaseDataObj

        def init_subclass(cls, **kwargs):
            super(BaseDataObj, cls).__init_subclass__(**kwargs)
            if 'save' in cls.__dict__:
                cls.save = _checking_save(cls.__dict__['save'])

        BaseDataObj.__init_subclass__ = classmethod(init_subclass)


def pytest_terminal_summary(terminalreporter):
    if _checked_classes:
        checked = sorted(_checked_classes)
        terminalreporter.write_line('Object formats checked: ' +
                                    ', '.join(f'{name}/{fmt}' for name, fmt in checked))
