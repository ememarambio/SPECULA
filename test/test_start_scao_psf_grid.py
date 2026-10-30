"""Check finite exposure accounting and deterministic, batched field replay."""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy.io import fits

import specula
if specula.global_precision is None:
    specula.init(-1)

from specula.base_value import BaseValue
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.psf_exposure_store import PsfExposureStore
from specula.processing_objects.windowed_integration import WindowedIntegration


ROOT = Path(__file__).resolve().parents[1]
MODULE = importlib.util.spec_from_file_location(
    'start_scao_psfs', ROOT / 'config/START-SCAO/run_psf_grid.py')
workflow = importlib.util.module_from_spec(MODULE)
MODULE.loader.exec_module(workflow)


def test_nonoverlapping_exposures_stream_with_exact_times(tmp_path):
    params = SimulParams(time_step=0.002)
    value = BaseValue(value=np.zeros((2, 2)), target_device_idx=-1, precision=1)
    objects = []
    stores = []
    for count in (100, 5000):
        window = WindowedIntegration(params, [2, 2], dt=count * 0.002,
                                     start_time=0.2, update_time_on_dt=True,
                                     target_device_idx=-1, precision=1)
        window.inputs['input'].set(value)
        store = PsfExposureStore(params, str(tmp_path / str(count)), 'J', 1250, 16.27,
                                 count, [[-10.5, 10.5]], [31], start_time=0.2)
        store.inputs['in_psfs'].set([window.output])
        objects.extend([window, store])
        stores.append(store)
    for obj in objects:
        obj.setup()
        obj.check_input_names()
    # 100 settling frames, 5000 science frames, then an incomplete 7-frame tail.
    for step in range(5107):
        t = value.seconds_to_t(step * 0.002)
        value.value[:] = -999 if step < 100 else (step - 100) // 100 + 1
        value.generation_time = t
        for obj in objects:
            if obj.check_ready(t):
                obj.trigger()
                obj.post_trigger()
    for obj in objects:
        obj.finalize()
    short_files = sorted((tmp_path / '100').glob('*.fits'))
    long_files = sorted((tmp_path / '5000').glob('*.fits'))
    assert len(short_files) == 50
    assert len(long_files) == 1
    for i, path in enumerate(short_files):
        with fits.open(path) as hdul:
            np.testing.assert_allclose(hdul[0].data, i + 1, rtol=2e-6)
            assert hdul[0].header['TSTART'] == pytest.approx(0.2 + i * 0.2)
            assert hdul[0].header['TEND'] == pytest.approx(0.4 + i * 0.2)
            assert hdul[0].header['NFRAMES'] == 100
            assert hdul['FIELD'].data['FIELD_ID'].tolist() == [31]
    np.testing.assert_allclose(fits.getdata(long_files[0]), 25.5, rtol=2e-5)
    assert fits.getheader(long_files[0])['EXPTIME'] == 10.0
    with pytest.raises(FileExistsError):
        stores[0].setup()


def test_grid_and_invalid_window_lengths():
    settings = yaml.safe_load((ROOT / 'config/START-SCAO/psf_grid.yml').read_text())
    points = workflow.field_grid(settings)
    assert points.shape == (1024, 2)
    np.testing.assert_allclose(points[[0, 31, 992, 1023]],
                               [[-10.5, -10.5], [10.5, -10.5], [-10.5, 10.5], [10.5, 10.5]])
    assert not np.any(np.all(points == 0, axis=1))
    params = {'main': {'time_step': 0.002, 'pixel_pupil': 256}}
    assert settings['nd'] == 2.0 and settings['stamp_size'] == 128
    assert workflow.validate_settings(settings, params) == (100, 5000)
    settings['stamp_size'] = 513
    with pytest.raises(ValueError, match='stamp_size'):
        workflow.validate_settings(settings, params)
    settings['stamp_size'] = 128
    settings['science_time'] = 9.8
    with pytest.raises(ValueError, match='whole short and long'):
        workflow.validate_settings(settings, params)


@pytest.mark.parametrize('band,wavelength', [('J', 1250.), ('K', 2200.)])
def test_128_pixel_stamp_preserves_center_sampling_and_flux(tmp_path, band, wavelength):
    from specula.lib.calc_psf import calc_psf_geometry

    params = SimulParams(time_step=0.002, pixel_pupil=256, pixel_pitch=0.0309462109375)
    geometry = calc_psf_geometry(params.pixel_pupil, params.pixel_pitch, wavelength, nd=2.)
    image = np.zeros((512, 512))
    image[256, 256] = 0.5
    image[255, 257] = 0.25
    image[0, 0] = 0.25  # Flux outside the stamp must not brighten the core.
    value = BaseValue(value=image, target_device_idx=-1)
    value.generation_time = 0
    store = PsfExposureStore(params, str(tmp_path), band, wavelength, geometry.pixel_size_mas,
                             1, [[0., 0.]], [-1], stamp_size=128)
    store.inputs['in_psfs'].set([value])
    store.setup()
    assert store.check_ready(0)
    store.trigger()
    with fits.open(tmp_path / 'exposure_000000.fits') as hdul:
        data, header = hdul[0].data, hdul[0].header
        assert data.shape == (1, 128, 128)
        np.testing.assert_array_equal(data[0], image[192:320, 192:320])
        assert data[0, 64, 64] == 0.5
        assert data.sum() == 0.75
        assert header['PIXSCALE'] == pytest.approx(geometry.pixel_size_mas)
        assert header['FOVX'] == pytest.approx(128 * geometry.pixel_size_mas / 1000)
        assert header['FOVY'] == header['FOVX']
        assert header['FULLNX'] == header['FULLNY'] == 512
        assert header['CROPX0'] == header['CROPY0'] == 192
        assert header['NORM'] == 'FULLPSF'
    oversized = PsfExposureStore(params, str(tmp_path / 'invalid'), band, wavelength,
                                 geometry.pixel_size_mas, 1, [[0., 0.]], [-1], stamp_size=513)
    oversized.inputs['in_psfs'].set([value])
    with pytest.raises(ValueError, match='stamp_size'):
        oversized.setup()


def test_batched_replay_matches_direct_field_psfs(tmp_path):
    # Small physical simulation with moving turbulence and a time-varying DM.
    # It exercises the one-step delayed DM input, all square corners, both
    # wavelengths, and a final batch smaller than batch_size.
    settings = dict(science_time=0.032, settling_time=0.004, short_frames=4,
                    long_exposure=0.016, field_width_arcsec=21., grid_size=2,
                    batch_size=3, nd=2., stamp_size=8, bands_nm={'J': 1250., 'K': 2200.})
    params = {
        'main': dict(class_='SimulParams', root_dir=str(tmp_path / 'calib'),
                     total_time=0.036, time_step=0.002, pixel_pupil=8, pixel_pitch=0.1),
        'seeing': dict(class_='WaveGenerator', constant=0.8),
        'wind': dict(class_='WaveGenerator', constant=[2.]),
        'direction': dict(class_='WaveGenerator', constant=[30.]),
        'atmo': dict(class_='AtmoEvolution', simul_params_ref='main', L0=20.,
                     heights=[1000.], Cn2=[1.], fov=21., pixel_phasescreens=128, seed=17,
                     inputs=dict(seeing='seeing.output', wind_speed='wind.output',
                                 wind_direction='direction.output')),
        'integrator': dict(class_='WaveGenerator', constant=[20., -30.], slope=[4000., 2000.]),
        'dm': dict(class_='DM', simul_params_ref='main', type_str='zernike', nmodes=2,
                   npixels=8, obsratio=0., height=0.,
                   inputs=dict(in_command='integrator.output')),
        'pupilstop': dict(class_='Pupilstop', simul_params_ref='main', obs_diam=0.),
        'prop': dict(class_='AtmoPropagation', simul_params_ref='main', source_dict_ref=[],
                     inputs=dict(atmo_layer_list=['atmo.layer_list'],
                                 common_layer_list=['pupilstop', 'dm.out_layer:-1'])),
        'telemetry': dict(class_='DataStore', store_dir=str(tmp_path), create_tn=False,
                          inputs=dict(input_list=['comm-integrator.output'])),
    }
    for config in params.values():
        config['class'] = config.pop('class_')
    points = workflow.field_grid(settings)
    # Compare the cropped replay to an independently saved full-size direct run.
    full_settings = dict(settings, stamp_size=None)
    original = workflow.batch_params(params, full_settings, points, range(4), tmp_path / 'direct')
    workflow.run_params(original, tmp_path / 'science.yml')
    (tmp_path / 'psf_grid.yml').write_text(yaml.safe_dump(settings))
    (tmp_path / 'SCIENCE_COMPLETE').touch()
    workflow.replay_field(argparse.Namespace(run_dir=tmp_path, first_point=0,
                                            point_count=None, batch_size=None))
    for first, last in [(0, 3), (3, 4)]:
        batch = tmp_path / 'field' / f'batch_{first:04d}_{last - 1:04d}'
        assert (batch / 'COMPLETE.json').exists()
        replay_config = yaml.safe_load((batch / 'replay.yml').read_text())
        assert 'integrator' not in replay_config  # Commands come from DataSource.
        assert replay_config['store_J_short']['stamp_size'] == 8
        for band in ('J', 'K'):
            for label, count in [('short', 4), ('long', 2)]:
                outputs = sorted((batch / band / label).glob('*.fits'))
                assert len(outputs) == count
                for path in outputs:
                    expected = fits.getdata(tmp_path / 'direct' / band / label / path.name)[first:last, 4:12, 4:12]
                    np.testing.assert_allclose(fits.getdata(path), expected, atol=2e-7, rtol=2e-6)
                    assert fits.getdata(path).shape == (last - first, 8, 8)
                    assert np.all(fits.getdata(path).sum(axis=(1, 2)) < 1)
                    with fits.open(path) as hdul:
                        assert hdul[0].header['WAVELEN'] == settings['bands_nm'][band]
                        assert hdul['FIELD'].data['FIELD_ID'].tolist() == list(range(first, last))
    j = fits.getdata(tmp_path / 'direct/J/short/exposure_000000.fits')
    k = fits.getdata(tmp_path / 'direct/K/short/exposure_000000.fits')
    assert not np.allclose(j, k)
    # A missing command must fail before replay can reuse a stale command.
    with fits.open(tmp_path / 'comm.fits', mode='update') as hdul:
        hdul[1].data[2] = hdul[1].data[1]
    with pytest.raises(ValueError, match='without gaps'):
        workflow.validate_commands(tmp_path, params)
