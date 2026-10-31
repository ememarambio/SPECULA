"""Check aligned telemetry, physical PSF targets, and actual WebDataset round trips."""

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np
import pytest
from astropy.io import fits
import yaml

import specula
if specula.global_precision is None:
    specula.init(-1)

from specula.base_value import BaseValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.pixels import Pixels
from specula.data_objects.simul_params import SimulParams
from specula.lib.calc_psf import calc_psf
from specula.processing_objects.ccd import CCD
from specula.processing_objects.scao_telemetry_store import ScaoTelemetryStore
from specula.simul import Simul


ROOT = Path(__file__).resolve().parents[1]
MODULE = importlib.util.spec_from_file_location(
    'start_scao_dataset', ROOT / 'config/START-SCAO/generate_webdataset.py')
dataset = importlib.util.module_from_spec(MODULE)
MODULE.loader.exec_module(dataset)
import run_psf_grid as workflow


def telemetry_inputs(path):
    params = SimulParams(time_step=.002, total_time=.404)
    store = ScaoTelemetryStore(params, str(path), start_time=.004, target_device_idx=-1)
    commands = BaseValue(value=np.zeros(2, dtype=np.float32), target_device_idx=-1)
    applied = BaseValue(value=np.zeros(2, dtype=np.float32), target_device_idx=-1)
    pixels = Pixels(3, 2, target_device_idx=-1, precision=1)
    ef = ElectricField(4, 4, .1, target_device_idx=-1, precision=1)
    ef.A[:] = 1
    ef.A[0, 0] = 0
    for name, obj in zip(store.inputs, (commands, applied, pixels, ef)):
        store.inputs[name].set(obj)
    store.setup()
    return store, commands, applied, pixels, ef


def step_capture(store, commands, applied, pixels, ef, step):
    tick = step * store.step
    commands.value[:] = step
    applied.value[:] = step - 1
    pixels.pixels[:] = step
    ef.phaseInNm[:] = step
    for obj in (commands, pixels, ef):
        obj.generation_time = tick
    applied.generation_time = tick - store.step if step else -1
    assert store.check_ready(tick)
    store.trigger()
    store.post_trigger()


def test_capture_exact_windows_and_owned_bounded_buffers(tmp_path):
    objects = telemetry_inputs(tmp_path / 'telemetry')
    store = objects[0]
    buffers_id = None
    for step in range(202):
        step_capture(*objects, step)
        if step == 2:
            buffers_id = {k: id(v) for k, v in store.buffers.items()}
        if step >= 2:
            assert {k: id(v) for k, v in store.buffers.items()} == buffers_id
    store.finalize()
    assert json.loads((store.store_dir / 'COMPLETE.json').read_text())['windows'] == 2
    for index in range(2):
        with np.load(store.store_dir / f'window_{index:06d}.npz') as data:
            steps = np.arange(2 + index * 100, 102 + index * 100)
            np.testing.assert_array_equal(data['time_ticks'], steps * store.step)
            np.testing.assert_array_equal(data['dm_command_time_ticks'], (steps - 1) * store.step)
            np.testing.assert_array_equal(data['dm_commands'][:, 0], steps - 1)
            np.testing.assert_array_equal(data['controller_commands'][:, 0], steps)
            np.testing.assert_array_equal(data['shwfs_frames'][:, 0, 0], steps)
            np.testing.assert_allclose(data['residual_phase'][:, 1, 1], steps * 2 * np.pi / 1250, rtol=2e-7)
            assert np.all(data['residual_phase'][:, 0, 0] == 0)
    with pytest.raises(FileExistsError):
        store.setup()


def test_start_scao_capture_preserves_closed_loop_order(tmp_path):
    params = yaml.safe_load((ROOT / 'config/START-SCAO/params_truthwfs_scao.yml').read_text())
    settings = yaml.safe_load((ROOT / 'config/START-SCAO/psf_grid.yml').read_text())
    workflow.add_dataset_capture(params, settings, tmp_path)
    order, _ = Simul('dummy.yml').build_trigger_order(params)
    assert order.index('prop') < order.index('detector_truth_sh') < order.index('integrator')
    assert order.index('integrator') < order.index('dataset_telemetry') < order.index('dm')
    assert params['prop']['inputs']['common_layer_list'][-1] == 'dm.out_layer:-1'


@pytest.mark.parametrize('problem', ['stale_frame', 'current_dm', 'gap', 'partial'])
def test_capture_rejects_unsynchronized_or_incomplete_streams(tmp_path, problem):
    objects = telemetry_inputs(tmp_path / 'telemetry')
    store, commands, applied, pixels, ef = objects
    step_capture(*objects, 2)
    if problem == 'partial':
        with pytest.raises(ValueError, match='Incomplete telemetry'):
            store.finalize()
    elif problem == 'gap':
        with pytest.raises(ValueError, match='Missing'):
            step_capture(*objects, 4)
    else:
        for obj in (commands, pixels, ef):
            obj.generation_time = 3 * store.step
        applied.generation_time = 2 * store.step
        if problem == 'stale_frame':
            pixels.generation_time = 2 * store.step
        else:
            applied.generation_time = 3 * store.step
        assert store.check_ready(3 * store.step)
        with pytest.raises(ValueError, match='current simulation step|before the current DM'):
            store.trigger()
    assert not (store.store_dir / 'COMPLETE.json').exists()


@pytest.fixture(scope='module')
def physical_run(tmp_path_factory):
    root = tmp_path_factory.mktemp('scao_dataset')
    settings = dict(science_time=.4, settling_time=.004, short_frames=100,
                    long_exposure=.4, field_width_arcsec=21., grid_size=2,
                    batch_size=3, nd=2., stamp_size=8, bands_nm={'J': 1250., 'K': 2200.})
    params = {
        'main': dict(class_='SimulParams', root_dir=str(root / 'calib'), total_time=.404,
                     time_step=.002, pixel_pupil=8, pixel_pitch=.1),
        'seeing': dict(class_='WaveGenerator', constant=.8),
        'wind': dict(class_='WaveGenerator', constant=[2.]),
        'direction': dict(class_='WaveGenerator', constant=[30.]),
        'atmo': dict(class_='AtmoEvolution', simul_params_ref='main', L0=20., heights=[1000.],
                     Cn2=[1.], fov=21., pixel_phasescreens=128, seed=17,
                     inputs=dict(seeing='seeing.output', wind_speed='wind.output', wind_direction='direction.output')),
        'integrator': dict(class_='WaveGenerator', constant=[20., -30.], slope=[400., 200.]),
        'dm': dict(class_='DM', simul_params_ref='main', type_str='zernike', nmodes=2,
                   npixels=8, obsratio=0., height=0., inputs=dict(in_command='integrator.output')),
        'pupilstop': dict(class_='Pupilstop', simul_params_ref='main', obs_diam=0.),
        'prop': dict(class_='AtmoPropagation', simul_params_ref='main',
                     source_dict_ref=['source_science_j', 'source_science_k', 'source_ngs'],
                     inputs=dict(atmo_layer_list=['atmo.layer_list'],
                                 common_layer_list=['pupilstop', 'dm.out_layer:-1'])),
        'truth_sh': dict(class_='SH', subap_on_diameter=2, subap_wanted_fov=2.4,
                         sensor_pxscale=.4, subap_npx=6, wavelengthInNm=700.,
                         inputs=dict(in_ef='prop.out_source_ngs_ef')),
        'detector_truth_sh': dict(class_='CCD', simul_params_ref='main', size=[12, 12], dt=.002,
                                  bandw=200., photon_noise=True, readout_noise=True, readout_level=1.6,
                                  inputs=dict(in_i='truth_sh.out_i')),
        'telemetry': dict(class_='DataStore', store_dir=str(root), create_tn=False, data_format='fits',
                          inputs=dict(input_list=['comm-integrator.output'])),
    }
    for name, band, wavelength in [('source_science_j', 'J', 1250.), ('source_science_k', 'K', 2200.),
                                   ('source_ngs', 'R', 700.)]:
        params[name] = dict(class_='Source', polar_coordinates=[0., 0.], magnitude=8.,
                            wavelengthInNm=wavelength, band=band)
    for band, wavelength in settings['bands_nm'].items():
        params[f'psf_{band.lower()}'] = dict(class_='PSF', simul_params_ref='main', wavelengthInNm=wavelength,
                                             nd=2., start_time=.004,
                                             inputs=dict(in_ef=f'prop.out_source_science_{band.lower()}_ef'))
    for config in params.values():
        config['class'] = config.pop('class_')
    workflow.add_dataset_capture(params, settings, root)
    # A known time-varying controller output lets us check the applied-command
    # offset independently of the phase/PSF checks, without external calibration.
    params['dataset_telemetry']['inputs']['in_commands'] = 'integrator.output'
    workflow.add_exposures(params, settings, {'J': ['psf_j'], 'K': ['psf_k']}, [[0., 0.]], [-1], root / 'on_axis')
    order, _ = Simul('dummy.yml').build_trigger_order(params)
    assert order.index('integrator') < order.index('dataset_telemetry') < order.index('dm')
    assert order.index('prop') < order.index('detector_truth_sh') < order.index('dataset_telemetry')
    (root / 'psf_grid.yml').write_text(yaml.safe_dump(settings))
    # An independent observer of the actual noisy detector output verifies that
    # every saved frame is the original exposure, not a shifted or regenerated one.
    detector_frames = []
    detector_trigger = CCD.trigger_code

    def observe_detector(ccd):
        detector_trigger(ccd)
        detector_frames.append(specula.cpuArray(ccd.outputs['out_pixels'].pixels, force_copy=True))

    with patch.object(CCD, 'trigger_code', observe_detector):
        workflow.run_params(params, root / 'science.yml')
    np.save(root / 'test_detector_frames.npy', detector_frames)
    (root / 'SCIENCE_COMPLETE').touch()
    workflow.replay_field(argparse.Namespace(run_dir=root, first_point=0, point_count=None, batch_size=None))
    return root


def test_saved_phases_reproduce_on_axis_exposure_and_command_delay(physical_run):
    run = dataset.RunDataset(physical_run)
    commands = fits.getdata(physical_run / 'comm.fits')
    frames = np.load(physical_run / 'test_detector_frames.npy')
    for index in range(run.count):
        with np.load(run.telemetry / f'window_{index:06d}.npz') as data:
            first = run.start_step + index * 100
            np.testing.assert_array_equal(data['dm_commands'], commands[first - 1:first + 99])
            np.testing.assert_array_equal(data['controller_commands'], commands[first:first + 100])
            np.testing.assert_array_equal(data['shwfs_frames'], frames[first:first + 100])
            for band, wavelength in run.settings['bands_nm'].items():
                phase = data['residual_phase'] * (1250. / wavelength)
                expected = np.mean([calc_psf(p, data['pupil'], imwidth=16, normalize=True,
                                             complex_dtype=np.complex128) for p in phase], axis=0)[4:12, 4:12]
                saved = fits.getdata(physical_run / 'on_axis' / band / 'short' / f'exposure_{index:06d}.fits')[0]
                np.testing.assert_allclose(saved, expected, rtol=3e-5, atol=2e-8)


def test_webdataset_roundtrip_and_shard_rotation(physical_run, tmp_path):
    wds = pytest.importorskip('webdataset')
    output = tmp_path / 'shards'
    manifest = dataset.export_dataset([physical_run], output, shard_size=1)
    assert manifest['samples'] == 2
    assert [s['samples'] for s in manifest['shards']] == [1, 1]
    urls = [str(output / shard['file']) for shard in manifest['shards']]
    samples = list(wds.WebDataset(urls, shardshuffle=False).decode())
    assert len(samples) == 2
    run = dataset.RunDataset(physical_run)
    for index, sample in enumerate(samples):
        with np.load(run.telemetry / f'window_{index:06d}.npz') as original:
            for name in original.files:
                np.testing.assert_array_equal(sample[f'{name}.npy'], original[name])
        np.testing.assert_array_equal(sample['field_ids.npy'], [-1, 0, 1, 2, 3])
        assert sample['psf_j.npy'].shape == sample['psf_k.npy'].shape == (5, 8, 8)
        assert sample['meta.json']['first_simulation_step'] == 2 + index * 100
        for directory, ids in run.groups:
            for band in ('J', 'K'):
                expected = fits.getdata(directory / band / 'short' / f'exposure_{index:06d}.fits')
                np.testing.assert_array_equal(sample[f'psf_{band.lower()}.npy'][ids + 1], expected)
    with pytest.raises(FileExistsError):
        dataset.export_dataset([physical_run], output)
    with pytest.raises(ValueError, match='duplicate'):
        dataset.export_dataset([physical_run, physical_run], tmp_path / 'duplicate')


@pytest.mark.parametrize('problem', ['missing_field', 'duplicate_field', 'incomplete_batch',
                                    'shifted_psf', 'wrong_band', 'wrong_coordinates',
                                    'shifted_telemetry', 'wrong_commands'])
def test_export_rejects_misaligned_data_without_publishing(physical_run, tmp_path, problem):
    pytest.importorskip('webdataset')
    run = tmp_path / 'run'
    shutil.copytree(physical_run, run)
    batch = run / 'field/batch_0003_0003'
    if problem == 'missing_field':
        shutil.rmtree(batch)
    elif problem == 'duplicate_field':
        shutil.copytree(batch, run / 'field/batch_duplicate')
    elif problem == 'incomplete_batch':
        (batch / 'COMPLETE.json').unlink()
    elif problem in ('shifted_psf', 'wrong_band', 'wrong_coordinates'):
        with fits.open(batch / 'J/short/exposure_000001.fits', mode='update') as hdul:
            if problem == 'shifted_psf':
                hdul[0].header['TSTART'] += .002
            elif problem == 'wrong_band':
                hdul[0].header['WAVELEN'] = 2200.
            else:
                hdul['FIELD'].data['X_ARCSEC'][0] += 1
    else:
        path = run / 'dataset_telemetry/window_000001.npz'
        with np.load(path) as original:
            arrays = dict(original)
        if problem == 'shifted_telemetry':
            arrays['time_ticks'] += 2_000_000
        else:
            arrays['controller_commands'][0] += 1
        np.savez(path, **arrays)
    with pytest.raises(ValueError):
        dataset.export_dataset([run], tmp_path / 'shards', shard_size=1)
    assert not (tmp_path / 'shards').exists()
    assert not list(tmp_path.glob('.shards.partial-*'))
