"""Run START-SCAO once, then replay its commands for windowed J/K field PSFs.

Run from the repository root. See README.md for calibration and output details.
"""

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import sys

import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))


def frame_count(seconds, step, name):
    if not np.isfinite(seconds):
        raise ValueError(f'{name} must be finite')
    count = round(seconds / step)
    if count < 0 or not np.isclose(
            count * step, seconds, rtol=0, atol=1e-10):
        raise ValueError(f'{name} must be a nonnegative integer multiple of time_step')
    return count


def validate_settings(settings, params):
    step = params['main']['time_step']
    if not np.isfinite(step) or step <= 0:
        raise ValueError('time_step must be positive')
    science = frame_count(settings['science_time'], step, 'science_time')
    frame_count(settings['settling_time'], step, 'settling_time')
    long = frame_count(settings['long_exposure'], step, 'long_exposure')
    short = settings['short_frames']
    if not isinstance(short, int) or short <= 0 or long <= 0:
        raise ValueError('Exposure frame counts must be positive integers')
    if science <= 0 or science % short or science % long:
        raise ValueError('science_time must contain whole short and long exposures')
    for key in ('grid_size', 'batch_size'):
        if not isinstance(settings[key], int) or settings[key] < (2 if key == 'grid_size' else 1):
            raise ValueError(f'Invalid {key}')
    for key in ('field_width_arcsec', 'nd'):
        if not np.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if settings['nd'] < 1:
        raise ValueError('nd must be at least 1')
    if set(settings['bands_nm']) != {'J', 'K'} or any(
            not np.isfinite(v) or v <= 0 for v in settings['bands_nm'].values()):
        raise ValueError('bands_nm must specify positive J and K wavelengths')
    return short, long


def field_grid(settings):
    axis = np.linspace(-settings['field_width_arcsec'] / 2,
                       settings['field_width_arcsec'] / 2, settings['grid_size'])
    x, y = np.meshgrid(axis, axis, indexing='xy')
    return np.column_stack((x.ravel(), y.ravel()))


def add_exposures(params, settings, psf_names, positions, field_ids, output_dir):
    """Attach two independent averaging windows and streaming sinks per band."""
    from specula.lib.calc_psf import calc_psf_geometry

    short, long = validate_settings(settings, params)
    main = params['main']
    size = int(np.around(main['pixel_pupil'] * settings['nd'] / 2) * 2)
    for band, wavelength in settings['bands_nm'].items():
        geometry = calc_psf_geometry(main['pixel_pupil'], main['pixel_pitch'],
                                     wavelength, nd=settings['nd'])
        for label, count in [('short', short), ('long', long)]:
            refs = []
            for i, psf in enumerate(psf_names[band]):
                name = f'exposure_{band}_{label}_{i}'
                params[name] = {
                    'class': 'WindowedIntegration', 'simul_params_ref': 'main',
                    'n_elem': [size, size], 'dt': count * main['time_step'],
                    'start_time': settings['settling_time'], 'update_time_on_dt': True,
                    'inputs': {'input': f'{psf}.out_psf'},
                }
                refs.append(f'{name}.output')
            params[f'store_{band}_{label}'] = {
                'class': 'PsfExposureStore', 'simul_params_ref': 'main',
                'store_dir': str(Path(output_dir) / band / label),
                'band': band, 'wavelength_nm': wavelength,
                'pixel_size_mas': geometry.pixel_size_mas, 'n_frames': count,
                'positions': np.asarray(positions).tolist(),
                'field_ids': list(map(int, field_ids)),
                'start_time': settings['settling_time'], 'inputs': {'in_psfs': refs},
            }


def run_params(params, path):
    from specula.simul import Simul

    path.write_text(yaml.safe_dump(params, sort_keys=False))
    # Always start at zero: both atmospheric evolution and the delayed DM
    # feedback must have exactly the same history as in the original run.
    Simul(str(path), log_level='warning').run()


def run_science(args):
    params = yaml.safe_load((HERE / 'params_truthwfs_scao.yml').read_text())
    settings = yaml.safe_load(args.settings.read_text())
    if args.science_time is not None:
        settings['science_time'] = args.science_time
    validate_settings(settings, params)
    params['main']['total_time'] = settings['settling_time'] + settings['science_time']
    params['main']['root_dir'] = str(Path(params['main']['root_dir']).resolve())
    params['atmo']['fov'] = settings['field_width_arcsec']
    for band, wavelength in settings['bands_nm'].items():
        params[f'psf_{band.lower()}'].update(
            wavelengthInNm=wavelength, nd=settings['nd'], start_time=settings['settling_time'])
        params[f'source_science_{band.lower()}']['wavelengthInNm'] = wavelength
    if args.displays:
        params.update(yaml.safe_load((HERE / 'display.yml').read_text()))
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    settings_path = run_dir / 'psf_grid.yml'
    settings_path.write_text(yaml.safe_dump(settings, sort_keys=False))
    params['telemetry'] = {
        'class': 'DataStore', 'store_dir': str(run_dir), 'create_tn': False,
        'inputs': {'input_list': ['comm-integrator.out_comm']},
    }
    add_exposures(params, settings, {'J': ['psf_j'], 'K': ['psf_k']},
                  [[0.0, 0.0]], [-1], run_dir / 'on_axis')
    run_params(params, run_dir / 'science.yml')
    validate_commands(run_dir, params)
    (run_dir / 'SCIENCE_COMPLETE').write_text('Full-cadence commands and on-axis exposures saved.\n')
    print(f'Science run complete: {run_dir}')


def validate_commands(run_dir, params):
    """Reject missing frames instead of replaying a stale DM command silently."""
    from astropy.io import fits
    from specula.base_time_obj import BaseTimeObj

    step = params['main']['time_step']
    count = frame_count(params['main']['total_time'], step, 'total_time')
    ticks = BaseTimeObj(target_device_idx=-1).seconds_to_t(step)
    # DataStore timestamps are uint64 FITS images with BZERO scaling.
    with fits.open(run_dir / 'comm.fits', memmap=False) as hdul:
        times = hdul[1].data
        if hdul[0].header.get('DOWNSAMP', 1) != 1 or not np.array_equal(
                times, np.arange(count, dtype=np.int64) * ticks):
            raise ValueError('Replay requires every DM command from t=0, without gaps or downsampling')


def batch_params(base, settings, positions, field_ids, output_dir):
    params = deepcopy(base)
    if params['prop'].get('doFresnel', False) or params['prop'].get('enable_chromatic_effect', False):
        raise ValueError('Sharing J/K OPD requires geometric propagation without chromatic shifts')
    sources, names = [], {'J': [], 'K': []}
    for i, (x, y) in enumerate(positions):
        source = f'field_{i}'
        sources.append(source)
        params[source] = {
            'class': 'Source', 'polar_coordinates': [float(np.hypot(x, y)),
                float(np.degrees(np.arctan2(y, x)) % 360)],
            'magnitude': 8.0, 'wavelengthInNm': settings['bands_nm']['J'], 'band': 'J',
        }
        for band, wavelength in settings['bands_nm'].items():
            name = f'psf_{band}_{i}'
            names[band].append(name)
            params[name] = {
                'class': 'PSF', 'simul_params_ref': 'main', 'wavelengthInNm': wavelength,
                'nd': settings['nd'], 'start_time': settings['settling_time'],
                'inputs': {'in_ef': f'prop.out_{source}_ef'},
            }
    params['prop']['source_dict_ref'] = sources
    params['prop']['outputs'] = [f'out_{s}_ef' for s in sources]
    add_exposures(params, settings, names, positions, field_ids, output_dir)
    return params


def replay_field(args):
    from specula.base_replay_analyser import BaseReplayAnalyser

    run_dir = args.run_dir.resolve()
    if not (run_dir / 'SCIENCE_COMPLETE').exists():
        raise ValueError('Run the science stage successfully before field replay')
    settings = yaml.safe_load((run_dir / 'psf_grid.yml').read_text())
    analyser = BaseReplayAnalyser(str(run_dir.parent), run_dir.name, start_time=0)
    validate_settings(settings, analyser.params)
    validate_commands(run_dir, analyser.params)
    if analyser.params['atmo']['fov'] < settings['field_width_arcsec']:
        raise ValueError('Recorded atmosphere is too small for the field; rerun science')
    # An optional uncorrected display propagation is not part of the science beam.
    analyser.params.pop('prop_uncorrected', None)
    base = analyser._build_replay_params_from_datastore('prop')
    points = field_grid(settings)
    start = args.first_point
    stop = len(points) if args.point_count is None else start + args.point_count
    batch_size = settings['batch_size'] if args.batch_size is None else args.batch_size
    if not 0 <= start < stop <= len(points) or batch_size <= 0:
        raise ValueError('Invalid field range or batch size')
    output = run_dir / 'field'
    output.mkdir(exist_ok=True)
    # Row-major grid: FIELD_ID = iy * grid_size + ix, x varies fastest.
    np.savetxt(output / 'grid.csv', np.column_stack((np.arange(len(points)), points)),
               delimiter=',', header='field_id,x_arcsec,y_arcsec', comments='',
               fmt=['%d', '%.12g', '%.12g'])
    for first in range(start, stop, batch_size):
        last = min(first + batch_size, stop)
        batch = output / f'batch_{first:04d}_{last - 1:04d}'
        batch.mkdir(exist_ok=False)
        print(f'Replaying field points {first}–{last - 1} of {len(points)}', flush=True)
        params = batch_params(base, settings, points[first:last], range(first, last), batch)
        run_params(params, batch / 'replay.yml')
        (batch / 'COMPLETE.json').write_text(json.dumps({'first': first, 'stop': last}) + '\n')
        gc.collect()
    print(f'Field PSFs saved in {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=int, default=-1, help='-1 for CPU, 0+ for CUDA GPU')
    parser.add_argument('--precision', type=int, choices=[0, 1], default=1)
    sub = parser.add_subparsers(dest='command', required=True)
    science = sub.add_parser('run', help='Run the AO loop and save commands/on-axis exposures')
    science.add_argument('run_dir', type=Path, help='New output directory')
    science.add_argument('--settings', type=Path, default=HERE / 'psf_grid.yml')
    science.add_argument('--science-time', type=float, help='Usable duration after settling [s]')
    science.add_argument('--displays', action='store_true')
    replay = sub.add_parser('replay', help='Compute J/K field exposures from a saved run')
    replay.add_argument('run_dir', type=Path)
    replay.add_argument('--batch-size', type=int)
    replay.add_argument('--first-point', type=int, default=0)
    replay.add_argument('--point-count', type=int)
    args = parser.parse_args()
    import specula
    precision = args.precision
    if args.command == 'replay':
        # Select precision before importing any SPECULA submodule: several
        # modules capture the defaults at import time, so re-init is too late.
        recorded = yaml.safe_load((args.run_dir / 'replay_params.yml').read_text())
        precision = recorded['data_source']['global_precision']
    specula.init(args.target, precision=precision)
    if args.command == 'run':
        run_science(args)
    else:
        replay_field(args)


if __name__ == '__main__':
    main()
