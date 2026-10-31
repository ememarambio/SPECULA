'''
Storage formats of DataStore and DataSource (see iteras.lib.storage).
'''
import specula
specula.init(0)  # Default target device

import os
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from iteras.lib import storage
from iteras.simul import Simul
from iteras.processing_objects.data_store import DataStore
from iteras.processing_objects.data_source import DataSource
from iteras.base_value import BaseValue


def _env_without_format():
    return {k: v for k, v in os.environ.items()
            if k not in ('ITERAS_DATA_FORMAT', 'SPECULA_DATA_FORMAT')}


class TestDefaultFormat(unittest.TestCase):

    def test_default_is_fits(self):
        with patch.dict(os.environ, _env_without_format(), clear=True):
            self.assertEqual(storage.default_format(), 'fits')
            self.assertEqual(DataStore(store_dir='unused').data_format, 'fits')

    def test_environment_variable(self):
        env = _env_without_format()
        env['ITERAS_DATA_FORMAT'] = 'npz'
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(storage.default_format(), 'npz')
            self.assertEqual(DataStore(store_dir='unused').data_format, 'npz')
            # An explicit format wins
            self.assertEqual(DataStore(store_dir='unused', data_format='pickle').data_format, 'pickle')

    def test_specula_environment_variable(self):
        env = _env_without_format()
        env['SPECULA_DATA_FORMAT'] = 'npz'
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(storage.default_format(), 'npz')
            # ITERAS_DATA_FORMAT wins if both are set
            os.environ['ITERAS_DATA_FORMAT'] = 'pickle'
            self.assertEqual(storage.default_format(), 'pickle')

    def test_invalid_formats(self):
        with patch.dict(os.environ, {'ITERAS_DATA_FORMAT': 'xls'}):
            with self.assertRaisesRegex(ValueError, 'ITERAS_DATA_FORMAT'):
                DataStore(store_dir='unused')
        with self.assertRaisesRegex(ValueError, 'xls'):
            DataStore(store_dir='unused', data_format='xls')
        with self.assertRaisesRegex(ValueError, 'xls'):
            DataSource(outputs=[], store_dir='unused', data_format='xls')


class TestSaveMethods(unittest.TestCase):
    '''DataStore methods saving the stored data in each format'''

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = DataStore(store_dir=self.tmp.name, create_tn=False)
        self.store.tn_dir = self.tmp.name
        self.store.local_inputs = {'v': BaseValue(value=np.zeros(2), target_device_idx=-1)}
        self.store.storage['v'][0] = np.zeros(2)
        self.store.storage['v'][1] = np.ones(2)

    def test_save_methods(self):
        for method, data_format in [(self.store.save_fits, 'fits'), (self.store.save_npz, 'npz'),
                                    (self.store.save_npy, 'npy'), (self.store.save_hdf5, 'hdf5')]:
            with self.subTest(data_format=data_format):
                method()
                path = storage.filename(self.tmp.name, 'v', data_format)
                with storage.read_object(path) as f:
                    np.testing.assert_array_equal(f.read('data', 0), [[0, 0], [1, 1]])
                    np.testing.assert_array_equal(f.read('times', 1), [0, 1])

    def test_errors_are_logged(self):
        '''An error saving one input is logged, and the other inputs are saved'''
        with patch.object(storage, 'write_object', side_effect=OSError('disk full')), \
                patch.object(self.store, 'logger') as logger:
            self.store.save_npz()
        self.assertIn('disk full', logger.error.call_args[0][0])


class TestRoundTrip(unittest.TestCase):
    '''
    Store a counter with DataStore in a short simulation, then read it back
    with DataSource, detecting the format from the files.
    '''

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.tmp.name
        pkg = os.path.join(root, 'storage_objs')
        os.makedirs(pkg)
        open(os.path.join(pkg, '__init__.py'), 'w').close()
        with open(os.path.join(pkg, 'counter.py'), 'w') as f:
            f.write(textwrap.dedent('''
                from iteras.base_processing_obj import BaseProcessingObj
                from iteras.base_value import BaseValue

                class Counter(BaseProcessingObj):
                    def __init__(self, target_device_idx=None, precision=None):
                        super().__init__(target_device_idx=target_device_idx, precision=precision)
                        self.out = BaseValue(value=self.xp.zeros(2, dtype=self.dtype),
                                             target_device_idx=target_device_idx)
                        self.outputs['out_value'] = self.out

                    def trigger_code(self):
                        self.out.value += self.xp.array([1, 10], dtype=self.dtype)

                    def post_trigger(self):
                        super().post_trigger()
                        self.out.generation_time = self.current_time
                '''))
        sys.path.insert(0, root)
        self.addCleanup(sys.path.remove, root)
        self.addCleanup(lambda: [sys.modules.pop(m) for m in list(sys.modules)
                                 if m.split('.')[0] == 'storage_objs'])

    def _store(self, data_format, downsample_factor=1):
        '''Run the simulation, return the store directory'''
        store_dir = os.path.join(self.tmp.name, f'store_{data_format}')
        store_pars = {'class': 'DataStore', 'store_dir': store_dir, 'create_tn': False,
                      'downsample_factor': downsample_factor,
                      'inputs': {'input_list': ['v-counter.out_value']}}
        if data_format is not None:
            store_pars['data_format'] = data_format
        params = {
            'main': {'class': 'SimParams', 'root_dir': self.tmp.name, 'total_time': 0.003,
                     'time_step': 0.001, 'add_modules': ['storage_objs']},
            'counter': {'class': 'Counter', 'target_device_idx': -1},
            'store': store_pars,
        }
        yml = os.path.join(self.tmp.name, 'params.yml')
        with open(yml, 'w') as f:
            yaml.dump(params, f)
        Simul(yml, speed_report=False).run()
        return store_dir

    def _check_round_trip(self, data_format, extension):
        store_dir = self._store(data_format)
        self.assertTrue(os.path.exists(os.path.join(store_dir, 'v' + extension)))
        self.assertEqual(storage.find_format(store_dir, 'v'), data_format)

        # The format actually used is recorded for the replay
        with open(os.path.join(store_dir, 'replay_params.yml')) as f:
            replay = yaml.safe_load(f)
        self.assertEqual(replay['data_source']['data_format'], data_format)

        source = DataSource(outputs=['v'], store_dir=store_dir)   # format detected
        self.assertIsInstance(source.outputs['v'], BaseValue)
        self.assertEqual(source.headers['v']['OBJ_TYPE'], 'BaseValue')
        self.assertEqual(int(source.headers['v']['DOWNSAMP']), 1)
        times = sorted(source.storage['v'])
        self.assertEqual(len(times), 3)
        # The counter is incremented at each step (the last value is stored
        # twice, once more by DataStore.finalize())
        np.testing.assert_allclose(source.storage['v'][times[0]], [1, 10])
        np.testing.assert_allclose(source.storage['v'][times[-1]], [3, 30])

    def test_round_trip_fits(self):
        self._check_round_trip('fits', '.fits')

    def test_round_trip_pickle(self):
        self._check_round_trip('pickle', '.pickle')

    def test_round_trip_npz(self):
        self._check_round_trip('npz', '.npz')

    def test_round_trip_npy(self):
        self._check_round_trip('npy', '.npy')

    def test_round_trip_hdf5(self):
        self._check_round_trip('hdf5', '.h5')

    def test_format_from_environment(self):
        with patch.dict(os.environ, {'ITERAS_DATA_FORMAT': 'npz'}):
            store_dir = self._store(data_format=None)
        self.assertEqual(storage.find_format(store_dir, 'v'), 'npz')

    def test_downsampled_npz_rejected_by_replay(self):
        from iteras.base_replay_analyser import BaseReplayAnalyser
        store_dir = self._store('npz', downsample_factor=2)
        replay_params = {'data_source': {'store_dir': store_dir, 'outputs': ['v']}}
        analyser = BaseReplayAnalyser.__new__(BaseReplayAnalyser)
        analyser.tn_dir = store_dir
        with self.assertRaisesRegex(ValueError, 'DOWNSAMP=2'):
            analyser._validate_replay_inputs_are_not_downsampled(replay_params)

    def test_unknown_format_not_checked_by_replay(self):
        '''Files of unknown formats are left to DataSource, which rejects them'''
        from iteras.base_replay_analyser import BaseReplayAnalyser
        store_dir = self._store('npz', downsample_factor=2)
        replay_params = {'data_source': {'store_dir': store_dir, 'outputs': ['v'],
                                         'data_format': 'xls'}}
        analyser = BaseReplayAnalyser.__new__(BaseReplayAnalyser)
        analyser.tn_dir = store_dir
        analyser._validate_replay_inputs_are_not_downsampled(replay_params)


if __name__ == '__main__':
    unittest.main()
