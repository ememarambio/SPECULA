"""Verify direct WebDataset generation, synchronized telemetry and PSF windows."""

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import yaml

import specula
if specula.global_precision is None:
    specula.init(-1)

from specula.base_value import BaseValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.pixels import Pixels
from specula.data_objects.simul_params import SimulParams
from specula.lib.calc_psf import calc_psf
from specula.processing_objects.atmo_propagation import AtmoPropagation
from specula.processing_objects.ccd import CCD
from specula.processing_objects.scao_webdataset_store import ScaoWebdatasetStore
from specula.simul import Simul


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'config/START-SCAO/run_psf_grid.py'
MODULE = importlib.util.spec_from_file_location('start_scao_dataset', SCRIPT)
workflow = importlib.util.module_from_spec(MODULE)
MODULE.loader.exec_module(workflow)


@pytest.fixture
def small_config(tmp_path):
    return physical_config(tmp_path)


def physical_config(root):
    settings = dict(science_time=.8, settling_time=.004, short_frames=100,
                    long_exposure=.4, field_width_arcsec=21., grid_size=2,
                    nd=2., stamp_size=8, bands_nm={'J': 1250., 'K': 2200.})
    params = {
        'main': dict(class_='SimulParams', root_dir=str(root / 'calib'), total_time=.804,
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
    return params, settings


@pytest.fixture(scope='module')
def physical_run(tmp_path_factory):
    pytest.importorskip('webdataset')
    root = tmp_path_factory.mktemp('scao_dataset')
    params, settings = physical_config(root)
    workflow.add_direct_dataset(params, settings, root, shard_size=1)
    params['dataset']['inputs']['in_commands'] = 'integrator.output'
    order, _ = Simul('dummy.yml').build_trigger_order(params)
    assert order.index('integrator') < order.index('dataset') < order.index('dm')
    assert order.index('exposure_K_short_4') < order.index('dataset')
    frames, phases, targets = [], [], []
    detector_trigger, propagation_post = CCD.trigger_code, AtmoPropagation.post_trigger

    def observe_detector(ccd):
        detector_trigger(ccd)
        frames.append(specula.cpuArray(ccd.outputs['out_pixels'].pixels, force_copy=True))

    def observe_propagation(prop):
        propagation_post(prop)
        # Compute an independent instantaneous PSF reference from every field's
        # actual propagated OPD, before any exposure accumulation or crop.
        fields = [prop.outputs[f'out_{name}_ef'] for name in
                  ['source_science_j'] + [f'field_{i}' for i in range(4)]]
        amplitude = specula.cpuArray(fields[0].A, force_copy=True)
        phase = specula.cpuArray(fields[0].phaseInNm, force_copy=True) * (2 * np.pi / 1250.)
        phase -= phase[amplitude != 0].mean(dtype=np.float64)
        phase[amplitude == 0] = 0
        phases.append(phase)
        targets.append(np.array([
            [calc_psf(specula.cpuArray(ef.phaseInNm) * (2 * np.pi / wavelength),
                      specula.cpuArray(ef.A), imwidth=16, normalize=True,
                      complex_dtype=np.complex128) for ef in fields]
            for wavelength in (1250., 2200.)]))

    with patch.object(CCD, 'trigger_code', observe_detector), \
         patch.object(AtmoPropagation, 'post_trigger', observe_propagation):
        workflow.run_params(params, root / 'science.yml')
    return root, np.asarray(frames), np.asarray(phases), np.asarray(targets)


def test_samples_preserve_actual_frames_commands_phases_and_full_field(physical_run):
    root, frames, phases, targets = physical_run
    manifest, samples = read_shards(root / 'dataset')
    assert manifest['samples'] == len(samples) == 4
    assert [s['samples'] for s in manifest['shards']] == [1] * 4
    assert not list((root / 'dataset').rglob('*.fits'))
    assert not list(root.rglob('*.npz'))
    assert not list(root.rglob('*.partial'))
    for index, sample in enumerate(samples):
        first = 2 + index * 100
        steps = np.arange(first, first + 100)
        np.testing.assert_array_equal(sample['time_ticks.npy'], steps * 2_000_000)
        np.testing.assert_array_equal(sample['dm_command_time_ticks.npy'], (steps - 1) * 2_000_000)
        # These known, time-varying commands independently verify the DM lag.
        # WaveGenerator arithmetic rounds before capture. Allow its float32
        # cancellation error near zero without relaxing the step-alignment test.
        calculation_dtype = np.float32 if sample['meta.json']['global_precision'] == 1 else np.float64
        command_atol = 64 * np.finfo(calculation_dtype).eps
        np.testing.assert_allclose(sample['controller_commands.npy'],
                                   [20., -30.] + steps[:, None] * .002 * [400., 200.],
                                   atol=command_atol)
        np.testing.assert_allclose(sample['dm_commands.npy'],
                                   [20., -30.] + (steps[:, None] - 1) * .002 * [400., 200.],
                                   atol=command_atol)
        np.testing.assert_array_equal(sample['shwfs_frames.npy'], frames[first:first + 100])
        np.testing.assert_allclose(sample['residual_phase.npy'], phases[first:first + 100], rtol=2e-7, atol=2e-7)
        np.testing.assert_array_equal(sample['field_ids.npy'], [-1, 0, 1, 2, 3])
        np.testing.assert_array_equal(sample['field_xy_arcsec.npy'],
                                   [[0., 0.], [-10.5, -10.5], [10.5, -10.5], [-10.5, 10.5], [10.5, 10.5]])
        for band, name in enumerate(('j', 'k')):
            expected = targets[first:first + 100, band].mean(axis=0)[:, 4:12, 4:12]
            np.testing.assert_allclose(sample[f'psf_{name}.npy'], expected, rtol=3e-5, atol=2e-8)
        assert sample['meta.json']['first_simulation_step'] == first
        assert sample['meta.json']['n_frames'] == 100
        assert sample['meta.json']['on_axis_index'] == 0
        assert sample['meta.json']['phase_piston_removed'] is True
        assert 'pupil-mean piston removed per frame' in sample['meta.json']['phase_convention']
        pupil = sample['pupil.npy'] != 0
        np.testing.assert_allclose(sample['residual_phase.npy'][:, pupil].mean(axis=1, dtype=np.float64), 0, atol=2e-7)
    long_manifest, long_samples = read_shards(root / 'dataset/long_exposures')
    assert long_manifest['samples'] == len(long_samples) == 2
    for index, sample in enumerate(long_samples):
        first = 2 + index * 200
        assert sample['meta.json']['n_frames'] == 200
        assert sample['meta.json']['start_tick'] == first * 2_000_000
        assert sample['meta.json']['sample_kind'] == 'long_psf'
        assert 'shwfs_frames.npy' not in sample
        for band, name in enumerate(('j', 'k')):
            expected = targets[first:first + 200, band].mean(axis=0)[:, 4:12, 4:12]
            np.testing.assert_allclose(sample[f'psf_{name}.npy'], expected, rtol=3e-5, atol=2e-8)
            short_mean = np.mean([s[f'psf_{name}.npy'] for s in samples[index * 2:index * 2 + 2]], axis=0)
            np.testing.assert_allclose(sample[f'psf_{name}.npy'], short_mean, rtol=3e-5, atol=2e-8)


def read_shards(root):
    import webdataset as wds
    manifest = json.loads((root / 'dataset.json').read_text())
    urls = [str(root / shard['file']) for shard in manifest['shards']]
    return manifest, list(wds.WebDataset(urls, shardshuffle=False).decode())



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



def direct_inputs(path, device=-1, pixel_pupil=4, stamp_size=4, pixel_pitch=.1):
    params = SimulParams(time_step=.002, total_time=.404, pixel_pupil=pixel_pupil, pixel_pitch=pixel_pitch)
    store = ScaoWebdatasetStore(params, str(path), grid_size=2, field_width_arcsec=21.,
                                bands_nm={'J': 1250., 'K': 2200.}, stamp_size=stamp_size,
                                start_time=.004, long_frames=200,
                                shard_size=1, target_device_idx=device)
    xp = store.xp
    commands = BaseValue(value=xp.zeros(2, dtype=xp.float32), target_device_idx=device)
    applied = BaseValue(value=xp.zeros(2, dtype=xp.float32), target_device_idx=device)
    pixels = Pixels(3, 2, target_device_idx=device, precision=1)
    ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, target_device_idx=device, precision=1)
    ef.A[:] = 1
    for name, obj in zip(('in_commands', 'in_applied_commands', 'in_pixels', 'in_ef'),
                          (commands, applied, pixels, ef)):
        store.inputs[name].set(obj)
    psfs = {}
    for label in ('short', 'long'):
        for band in ('J', 'K'):
            key = f'in_{label}_{band}'
            size = 2 * pixel_pupil
            psfs[key] = [BaseValue(value=xp.ones((size, size), dtype=xp.float32) / size**2,
                                   target_device_idx=device) for _ in range(5)]
            store.inputs[key].set(psfs[key])
    return store, commands, applied, pixels, ef, psfs


@pytest.mark.parametrize('device', [-1, pytest.param(0, marks=pytest.mark.skipif(
    specula.cp is None, reason='CuPy/GPU unavailable'))])
def test_direct_writer_only_transfers_completed_psf_stamps(tmp_path, device):
    pytest.importorskip('webdataset')
    import specula.processing_objects.scao_webdataset_store as direct_module
    *objects, psfs = direct_inputs(tmp_path / 'dataset', device)
    store = objects[0]
    transfers = []
    psf_members = store._psf_members

    def observe_psf_transfers(label):
        with patch.object(direct_module, 'cpuArray', wraps=specula.cpuArray) as to_cpu:
            result = psf_members(label)
            transfers.extend(to_cpu.call_args_list)
        return result

    with patch.object(BaseValue, 'copyTo', side_effect=AssertionError('Full input clone')), \
         patch.object(BaseValue, 'transferDataTo', side_effect=AssertionError('Full input transfer')), \
         patch.object(store, '_psf_members', side_effect=observe_psf_transfers):
        store.setup()
        for step in range(202):
            if step in (101, 201):
                for key, values in psfs.items():
                    if 'short' in key or step == 201:
                        for value in values:
                            value.generation_time = step * store.step
            step_capture(*objects, step)
            assert len(transfers) == (0 if step < 101 else 10 if step < 201 else 30)
        assert all(call.args[0].shape == (4, 4) for call in transfers)
        store.finalize()
        assert len(transfers) == 30
    _, samples = read_shards(store.store_dir)
    assert np.allclose(samples[0]['psf_j.npy'].sum(axis=(1, 2)), .25)  # No crop renormalization.


@pytest.mark.parametrize('problem', ['stale_psf', 'partial', 'write_failure'])
def test_direct_failure_does_not_publish_dataset_or_open_shard(tmp_path, problem):
    pytest.importorskip('webdataset')
    *objects, psfs = direct_inputs(tmp_path / 'dataset')
    store = objects[0]
    store.setup()
    try:
        for step in range(101):
            step_capture(*objects, step)
        if problem == 'partial':
            with pytest.raises(ValueError, match='Incomplete telemetry'):
                store.finalize()
        elif problem == 'stale_psf':
            with pytest.raises(ValueError, match='PSFs do not match'):
                step_capture(*objects, 101)
        else:
            for key, values in psfs.items():
                if 'short' in key:
                    for value in values:
                        value.generation_time = 101 * store.step
            with patch.object(store.streams['short'].writer, 'write', side_effect=OSError('disk full')):
                with pytest.raises(OSError, match='disk full'):
                    step_capture(*objects, 101)
    finally:
        store.abort()
    assert all(stream.closed for stream in store.streams.values())
    assert not list(store.store_dir.rglob('dataset.json'))
    assert not list(store.store_dir.rglob('*.tar'))
    assert list(store.store_dir.rglob('*.partial'))


@pytest.mark.parametrize('interrupt', [False, True])
def test_run_entrypoint_and_interruption_cleanup(small_config, tmp_path, monkeypatch, interrupt):
    pytest.importorskip('webdataset')
    params, settings = small_config
    params['drive'] = params['integrator']
    params['integrator'] = {'class': 'Integrator', 'delay': 2, 'int_gain': [.3],
                            'n_modes': [2], 'inputs': {'delta_comm': 'drive.output'}}
    params['dm']['inputs']['in_command'] = 'integrator.out_comm'
    config = tmp_path / 'config'
    config.mkdir()
    (config / 'params_truthwfs_scao.yml').write_text(yaml.safe_dump(params))
    (config / 'psf_grid.yml').write_text(yaml.safe_dump(settings))
    monkeypatch.setattr(workflow, 'HERE', config)
    run = tmp_path / 'run'
    args = argparse.Namespace(run_dir=run, settings=config / 'psf_grid.yml', science_time=None,
                              displays=False, shard_size=1, max_shard_gib=1.)
    captured = []
    if interrupt:
        trigger = ScaoWebdatasetStore.trigger_code

        def fail_during_simulation(store):
            if store.current_time >= store.start + store.step:
                captured.append(store)
                raise RuntimeError('simulated interruption')
            trigger(store)

        monkeypatch.setattr(ScaoWebdatasetStore, 'trigger_code', fail_during_simulation)
        with pytest.raises(RuntimeError, match='simulated interruption'):
            workflow.run_science(args)
        assert captured and all(stream.closed for stream in captured[0].streams.values())
        assert not (run / 'DATASET_COMPLETE').exists()
        assert not list(run.rglob('dataset.json'))
        assert list(run.rglob('*.partial'))
    else:
        workflow.run_science(args)
        assert (run / 'DATASET_COMPLETE').exists()
        manifest, samples = read_shards(run / 'dataset')
        assert manifest['samples'] == len(samples) == 4
        for previous, current in zip(samples, samples[1:]):
            np.testing.assert_array_equal(current['dm_commands.npy'][0], previous['controller_commands.npy'][-1])
        with pytest.raises(FileExistsError):
            workflow.run_science(args)
    assert not list(run.rglob('*.fits'))
    assert not list(run.rglob('*.npz'))


def test_grid_and_invalid_settings():
    settings = yaml.safe_load((ROOT / 'config/START-SCAO/psf_grid.yml').read_text())
    params = {'main': {'time_step': .002, 'pixel_pupil': 256}}
    assert workflow.validate_settings(settings, params) == (100, 5000)
    points = workflow.field_grid(settings)
    assert points.shape == (1024, 2)
    np.testing.assert_array_equal(points[[0, 31, 992, 1023]],
                                 [[-10.5, -10.5], [10.5, -10.5], [-10.5, 10.5], [10.5, 10.5]])
    assert not np.any(np.all(points == 0, axis=1))
    for key, value, message in [('short_frames', 50, 'short_frames'),
                                 ('stamp_size', 513, 'stamp_size'),
                                 ('science_time', 9.8, 'whole short and long'),
                                 ('grid_size', 1, 'grid_size')]:
        invalid = {**settings, key: value}
        with pytest.raises(ValueError, match=message):
            workflow.validate_settings(invalid, params)


def test_observation_metadata_is_saved_in_short_and_long_shards(small_config, tmp_path):
    pytest.importorskip('webdataset')
    params, settings = small_config
    run = tmp_path / 'observation_metadata'
    workflow.add_direct_dataset(params, settings, run)
    params['dataset']['inputs']['in_commands'] = 'integrator.output'
    observation = {'observation_id': 'reference_profile_000123',
                   'random_seeds': {'atmosphere': 1001968}, 'profile_ids': ['good']}
    params['dataset']['observation_metadata'] = observation
    run.mkdir()
    workflow.run_params(params, run / 'science.yml')
    for directory in (run / 'dataset', run / 'dataset/long_exposures'):
        _, samples = read_shards(directory)
        assert samples
        for sample in samples:
            assert sample['meta.json']['observation'] == observation
            assert 'split' not in sample['meta.json']


def test_configured_closed_loop_order_and_full_grid(tmp_path):
    params = yaml.safe_load((ROOT / 'config/START-SCAO/params_truthwfs_scao.yml').read_text())
    settings = yaml.safe_load((ROOT / 'config/START-SCAO/psf_grid.yml').read_text())
    small = deepcopy(params)
    workflow.add_direct_dataset(params, settings, tmp_path)
    workflow.add_direct_dataset(small, {**settings, 'grid_size': 2}, tmp_path)
    order, _ = Simul('dummy.yml').build_trigger_order(small)
    assert order.index('prop') < order.index('detector_truth_sh') < order.index('integrator')
    assert order.index('integrator') < order.index('dataset') < order.index('dm')
    for band in ('J', 'K'):
        refs = params['dataset']['inputs'][f'in_short_{band}']
        assert len(refs) == 1025
        assert params[f'exposure_{band}_short_0']['inputs']['input'] == f'psf_{band.lower()}.out_psf'
        assert params[f'psf_{band}_0']['inputs']['in_ef'] == 'prop.out_field_0_ef'
        assert params[f'psf_{band}_0']['class'] == 'DatasetPSF'
        assert params[f'psf_{band}_0']['compute_statistics'] is False
        assert params[f'exposure_{band}_short_0']['stamp_size'] == 128
        assert params[f'exposure_{band}_long_0']['class'] == 'CompletedExposureIntegration'
        assert params[f'exposure_{band}_long_0']['inputs']['input'] == f'exposure_{band}_short_0.output'
    for option in ('psfs_are_stamps', 'device_buffering', 'background_writes'):
        assert params['dataset'][option] is True
    assert params['psf_j']['compute_statistics'] is False
    small['display_strehl'] = {'class': 'PlotDisplay'}
    workflow.add_direct_dataset(small, {**settings, 'grid_size': 2}, tmp_path)
    assert small['psf_j']['compute_statistics'] is True


@pytest.mark.parametrize('problem', ['stale_frame', 'current_dm', 'gap'])
def test_capture_rejects_unsynchronized_streams(tmp_path, problem):
    pytest.importorskip('webdataset')
    *objects, _ = direct_inputs(tmp_path / 'dataset')
    store, commands, applied, pixels, ef = objects
    store.setup()
    try:
        step_capture(*objects, 2)
        if problem == 'gap':
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
    finally:
        store.abort()
    assert not (store.store_dir / 'dataset.json').exists()


def test_native_128_pixel_stamp_preserves_center_scale_and_flux(tmp_path):
    pytest.importorskip('webdataset')
    *objects, psfs = direct_inputs(tmp_path / 'dataset', pixel_pupil=256, stamp_size=128,
                                   pixel_pitch=.0309462109375)
    store = objects[0]
    image = np.zeros((512, 512), dtype=np.float32)
    image[256, 256], image[255, 257], image[0, 0] = .5, .25, .25
    for values in psfs.values():
        for value in values:
            value.value[:] = image
    store.setup()
    buffers_id = None
    for step in range(202):
        if step in (101, 201):
            for key, values in psfs.items():
                if 'short' in key or step == 201:
                    for value in values:
                        value.generation_time = step * store.step
        step_capture(*objects, step)
        if step == 2:
            buffers_id = {k: id(v) for k, v in store.buffers.items()}
        if step >= 2:
            assert {k: id(v) for k, v in store.buffers.items()} == buffers_id
    store.finalize()
    _, samples = read_shards(store.store_dir)
    for band in ('j', 'k'):
        data = samples[0][f'psf_{band}.npy']
        assert data.shape == (5, 128, 128)
        np.testing.assert_array_equal(data[0], image[192:320, 192:320])
        assert data[0, 64, 64] == .5
        assert data[0].sum() == .75
    metadata = samples[0]['meta.json']
    assert metadata['crop_x0'] == metadata['crop_y0'] == 192
    assert metadata['pixel_scale_mas']['J'] == pytest.approx(16.273, rel=1e-4)
    assert metadata['pixel_scale_mas']['K'] == pytest.approx(28.640, rel=1e-4)


def test_only_generation_command_is_run():
    result = subprocess.run([sys.executable, str(SCRIPT), 'run', '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert '--dataset' not in result.stdout
    assert '--batch-size' not in result.stdout
    result = subprocess.run([sys.executable, str(SCRIPT), 'replay', 'unused'], capture_output=True, text=True)
    assert result.returncode == 2
    assert 'invalid choice' in result.stderr


@pytest.mark.parametrize('device', [-1, pytest.param(0, marks=pytest.mark.skipif(
    specula.cp is None, reason='CuPy/GPU unavailable'))])
@pytest.mark.parametrize('precision', [0, 1])
@pytest.mark.parametrize('device_buffering', [False, True])
@pytest.mark.parametrize('background_writes', [False, True])
def test_saved_residuals_remove_piston_per_frame(tmp_path, device, precision,
                                                device_buffering, background_writes):
    pytest.importorskip('webdataset')
    *objects, psfs = direct_inputs(tmp_path / 'dataset', device)
    store, commands, applied, pixels, _ = objects
    ef = ElectricField(4, 4, .1, target_device_idx=device, precision=precision)
    objects[-1] = ef
    store.inputs['in_ef'].set(ef)
    store.device_buffering = device_buffering
    store.background_writes = background_writes
    # An asymmetric, disconnected pupil makes averaging the whole stamp wrong.
    pupil = np.array([[1., .5, 0., 0.], [1., 1., 0., 1.],
                      [0., 0., 0., 1.], [1., 1., 0., 1.]])
    mask = pupil != 0
    ef.A[:] = store.xp.asarray(pupil)
    store.setup()
    expected, original_radians = [], []
    try:
        for step in range(202):
            tick = step * store.step
            commands.value[:] = step
            applied.value[:] = step - 1
            pixels.pixels[:] = step
            opd = (np.arange(16).reshape(4, 4) - 8) * (1 + .01 * step) + 400 + 5 * step
            opd[~mask] = 1e6  # Outside-pupil values must not contribute to piston.
            ef.phaseInNm[:] = store.xp.asarray(opd)
            original = specula.cpuArray(ef.phaseInNm).copy()
            for obj in (commands, pixels, ef):
                obj.generation_time = tick
            applied.generation_time = tick - store.step if step else -1
            if step in (101, 201):
                for key, values in psfs.items():
                    if 'short' in key or step == 201:
                        for value in values:
                            value.generation_time = tick
            assert store.check_ready(tick)
            store.trigger()
            store.post_trigger()
            np.testing.assert_array_equal(specula.cpuArray(ef.phaseInNm), original)
            if step >= 2:
                radians = original.astype(np.float64) * (2 * np.pi / 1250.)
                original_radians.append(radians.copy())
                centered = radians.astype(np.float64)
                centered -= centered[mask].mean()
                centered[~mask] = 0
                expected.append(centered)
        store.finalize()
    finally:
        store.abort()
    _, samples = read_shards(store.store_dir)
    assert len(samples) == 2
    for index, sample in enumerate(samples):
        phase = sample['residual_phase.npy']
        assert phase.dtype == ef.phaseInNm.dtype
        np.testing.assert_allclose(phase, expected[index * 100:(index + 1) * 100], atol=1e-7, rtol=1e-6)
        np.testing.assert_allclose(phase[:, mask].mean(axis=1, dtype=np.float64), 0, atol=1e-7)
        np.testing.assert_array_equal(phase[:, ~mask], 0)
        assert sample['meta.json']['phase_piston_removed'] is True
        for band, wavelength in (('j', 1250.), ('k', 2200.)):
            # Removing only a global phase offset must preserve both band PSFs.
            for local_step in (0, 99):
                original = original_radians[index * 100 + local_step].astype(np.float64)
                before = calc_psf(original * 1250. / wavelength, pupil, imwidth=8,
                                  normalize=True, complex_dtype=np.complex128)
                after = calc_psf(phase[local_step].astype(np.float64) * 1250. / wavelength,
                                 pupil, imwidth=8, normalize=True, complex_dtype=np.complex128)
                np.testing.assert_allclose(after, before, rtol=3e-5, atol=2e-8)
    assert json.loads((store.store_dir / 'metadata.json').read_text())['phase_piston_removed'] is True


def test_constant_piston_is_zero_and_empty_pupil_is_rejected(tmp_path):
    store, *_ = direct_inputs(tmp_path / 'dataset')
    store.pupil = np.ones((4, 4))
    opd = np.full((4, 4), 100.)
    store._prepare_residual_phase(opd)
    np.testing.assert_array_equal(opd, 0)
    store.pupil[:] = 0
    with pytest.raises(ValueError, match='nonempty pupil'):
        store._prepare_residual_phase(np.ones((4, 4)))
