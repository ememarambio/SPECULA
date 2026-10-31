"""Run START-SCAO and stream full-field J/K training samples directly to WebDataset.

Run from the repository root. See README.md for calibration and output details.
"""

import argparse
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
    if not isinstance(short, int) or short != 100:
        raise ValueError('short_frames must be 100 for aligned training samples')
    if long <= 0:
        raise ValueError('long_exposure must contain at least one simulation step')
    if science <= 0 or science % short or science % long:
        raise ValueError('science_time must contain whole short and long exposures')
    if not isinstance(settings['grid_size'], int) or settings['grid_size'] < 2:
        raise ValueError('grid_size must be an integer >= 2')
    for key in ('field_width_arcsec', 'nd'):
        if not np.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if settings['nd'] < 1:
        raise ValueError('nd must be at least 1')
    stamp_size = settings.get('stamp_size')
    if stamp_size is not None:
        full_size = int(np.around(params['main']['pixel_pupil'] * settings['nd'] / 2) * 2)
        if not isinstance(stamp_size, int) or not 1 <= stamp_size <= full_size:
            raise ValueError('stamp_size must be a positive integer no larger than the full PSF')
    if set(settings['bands_nm']) != {'J', 'K'} or any(
            not np.isfinite(v) or v <= 0 for v in settings['bands_nm'].values()):
        raise ValueError('bands_nm must specify positive J and K wavelengths')
    return short, long


def field_grid(settings):
    axis = np.linspace(-settings['field_width_arcsec'] / 2,
                       settings['field_width_arcsec'] / 2, settings['grid_size'])
    x, y = np.meshgrid(axis, axis, indexing='xy')
    return np.column_stack((x.ravel(), y.ravel()))


def add_psf_windows(params, settings, psf_names):
    """Attach independent short and long intensity averages for the writer."""
    short, long = validate_settings(settings, params)
    main = params['main']
    size = int(np.around(main['pixel_pupil'] * settings['nd'] / 2) * 2)
    stamp = settings.get('stamp_size') or size
    windows = {}
    for band in ('J', 'K'):
        for label, count in [('short', short), ('long', long)]:
            refs = []
            for i, psf in enumerate(psf_names[band]):
                name = f'exposure_{band}_{label}_{i}'
                params[name] = {
                    'class': 'PsfStampIntegration', 'simul_params_ref': 'main',
                    'full_size': size, 'stamp_size': stamp, 'dt': count * main['time_step'],
                    'start_time': settings['settling_time'],
                    'inputs': {'input': f'{psf}.out_psf'},
                }
                refs.append(f'{name}.output')
            windows[f'in_{label}_{band}'] = refs
    return windows


def run_params(params, path):
    from specula.simul import Simul

    path.write_text(yaml.safe_dump(params, sort_keys=False))
    # Start the AO loop at zero, including settling and the delayed DM feedback.
    simulation = Simul(str(path), log_level='warning')
    try:
        simulation.run()
    except BaseException:
        # Interrupted simulations do not call finalize(). Close direct writers
        # without marking their open shards or their dataset as complete.
        for obj in getattr(simulation, 'objs', {}).values():
            if hasattr(obj, 'abort'):
                obj.abort()
        raise


def validate_science_graph(params):
    """Require the shared on-axis residual and synchronized detector/DM timing."""
    if params['detector_truth_sh']['dt'] != params['main']['time_step']:
        raise ValueError('Dataset capture requires one SHWFS exposure per simulation step')
    prop = params['prop']
    if (prop.get('doFresnel', False) or prop.get('enable_chromatic_effect', False)
            or 'dm.out_layer:-1' not in prop['inputs']['common_layer_list']):
        raise ValueError('Dataset capture requires geometric propagation with dm.out_layer:-1')
    for source in ('source_science_j', 'source_ngs'):
        if params[source]['polar_coordinates'][0] != 0:
            raise ValueError('Dataset capture requires on-axis science and SHWFS sources')


def add_direct_dataset(params, settings, run_dir, shard_size=4, max_shard_bytes=1024**3):
    """Compute the entire field in the live AO loop and write complete samples."""
    validate_settings(settings, params)
    validate_science_graph(params)
    points = field_grid(settings)
    sources, names = [], {'J': ['psf_j'], 'K': ['psf_k']}
    for name in ('psf_j', 'psf_k'):
        params[name]['class'] = 'DatasetPSF'
    for i, (x, y) in enumerate(points):
        source = f'field_{i}'
        sources.append(source)
        params[source] = {
            'class': 'Source', 'polar_coordinates': [float(np.hypot(x, y)),
                float(np.degrees(np.arctan2(y, x)) % 360)],
            'magnitude': 8., 'wavelengthInNm': settings['bands_nm']['J'], 'band': 'J',
        }
        for band, wavelength in settings['bands_nm'].items():
            name = f'psf_{band}_{i}'
            names[band].append(name)
            params[name] = {
                'class': 'DatasetPSF', 'simul_params_ref': 'main', 'wavelengthInNm': wavelength,
                'nd': settings['nd'], 'start_time': settings['settling_time'],
                'compute_profile_metrics': False,
                'inputs': {'in_ef': f'prop.out_{source}_ef'},
            }
    params['prop']['source_dict_ref'].extend(sources)
    params['prop'].setdefault('outputs', []).extend(f'out_{source}_ef' for source in sources)
    windows = add_psf_windows(params, settings, names)
    params['dataset'] = {
        'class': 'ScaoWebdatasetStore', 'simul_params_ref': 'main',
        'store_dir': str(Path(run_dir) / 'dataset'),
        'grid_size': settings['grid_size'], 'field_width_arcsec': settings['field_width_arcsec'],
        'bands_nm': settings['bands_nm'], 'nd': settings['nd'],
        'stamp_size': settings.get('stamp_size'), 'n_frames': settings['short_frames'],
        'psfs_are_stamps': True,
        'long_frames': frame_count(settings['long_exposure'], params['main']['time_step'], 'long_exposure'),
        'start_time': settings['settling_time'], 'dm_sign': params['dm'].get('sign', -1),
        'shard_size': shard_size, 'max_shard_bytes': max_shard_bytes,
        'inputs': {
            'in_commands': 'integrator.out_comm',
            'in_applied_commands': 'dm.out_clipped_command:-1',
            'in_pixels': 'detector_truth_sh.out_pixels',
            'in_ef': 'prop.out_source_science_j_ef',
            **windows,
        },
    }


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
    try:
        import webdataset  # Fail before creating a run or allocating the full GPU graph.
    except ImportError as exc:
        raise ImportError('Install dataset support: python -m pip install -e ".[dataset]"') from exc
    if (args.shard_size < 1 or not np.isfinite(args.max_shard_gib) or args.max_shard_gib <= 0):
        raise ValueError('Shard limits must be positive and finite')
    add_direct_dataset(params, settings, run_dir, args.shard_size, int(args.max_shard_gib * 1024**3))
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'psf_grid.yml').write_text(yaml.safe_dump(settings, sort_keys=False))
    run_params(params, run_dir / 'science.yml')
    (run_dir / 'DATASET_COMPLETE').write_text('Training and long-exposure WebDataset streams saved.\n')
    print(f'Dataset complete: {run_dir / "dataset"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=int, default=-1, help='-1 for CPU, 0+ for CUDA GPU')
    parser.add_argument('--precision', type=int, choices=[0, 1], default=1)
    sub = parser.add_subparsers(dest='command', required=True)
    science = sub.add_parser('run', help='Generate full-field training samples directly in WebDataset')
    science.add_argument('run_dir', type=Path, help='New output directory')
    science.add_argument('--settings', type=Path, default=HERE / 'psf_grid.yml')
    science.add_argument('--science-time', type=float, help='Usable duration after settling [s]')
    science.add_argument('--displays', action='store_true')
    science.add_argument('--shard-size', type=int, default=4, help='Dataset samples per shard (default: 4)')
    science.add_argument('--max-shard-gib', type=float, default=1.,
                         help='Dataset shard size threshold in GiB (default: 1; rotation between samples)')
    args = parser.parse_args()
    import specula
    specula.init(args.target, precision=args.precision)
    run_science(args)


if __name__ == '__main__':
    main()
