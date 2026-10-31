"""Join aligned START-SCAO telemetry and J/K exposure targets into WebDataset shards.

Run science with ``run_psf_grid.py run RUN --dataset``, finish the field replay,
then run ``generate_webdataset.py RUN --out shards/run01``. No simulator or GPU
is needed for this export. Existing output directories are never overwritten.
"""

import argparse
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
from astropy.io import fits
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_psf_grid import field_grid, frame_count, validate_settings


def npy_bytes(array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


class RunDataset:
    """Validate a complete run, then assemble only one sample at a time."""

    def __init__(self, run_dir):
        self.root = Path(run_dir).resolve()
        self.telemetry = self.root / 'dataset_telemetry'
        if not (self.root / 'SCIENCE_COMPLETE').is_file():
            raise ValueError(f'{self.root}: science run is incomplete')
        if not (self.telemetry / 'COMPLETE.json').is_file():
            raise ValueError(f'{self.root}: missing complete dataset telemetry; rerun science with --dataset')
        self.params = yaml.safe_load((self.root / 'params.yml').read_text())
        self.settings = yaml.safe_load((self.root / 'psf_grid.yml').read_text())
        validate_settings(self.settings, self.params)
        self.meta = json.loads((self.telemetry / 'metadata.json').read_text())
        complete = json.loads((self.telemetry / 'COMPLETE.json').read_text())
        self.run_id = self.meta['run_id']
        if (len(self.run_id) != 32 or any(c not in '0123456789abcdef' for c in self.run_id)
                or self.meta['schema_version'] != 1 or self.meta['n_frames'] != 100
                or self.settings['short_frames'] != 100 or complete['run_id'] != self.run_id):
            raise ValueError('Unsupported or inconsistent dataset telemetry metadata')
        self.dt = self.params['main']['time_step']
        self.step = round(self.dt * 1e9)
        self.start_step = frame_count(self.settings['settling_time'], self.dt, 'settling_time')
        self.science_steps = frame_count(self.settings['science_time'], self.dt, 'science_time')
        self.total_steps = self.start_step + self.science_steps
        self.count = self.science_steps // 100
        expected_meta = {
            'start_tick': self.start_step * self.step, 'end_tick': self.total_steps * self.step,
            'step_ticks': self.step, 'tick_seconds': 1e-9,
            'phase_wavelength_nm': self.settings['bands_nm']['J'], 'phase_unit': 'rad',
            'phase_direction_arcsec': [0., 0.], 'applied_command_delay_steps': 1,
        }
        if (any(self.meta.get(k) != v for k, v in expected_meta.items())
                or complete['windows'] != self.count
                or frame_count(self.params['main']['total_time'], self.dt, 'total_time') != self.total_steps):
            raise ValueError('Telemetry timing or phase metadata does not match the science run')
        self._check_files(self.telemetry, 'window', '.npz')
        self.positions = np.vstack(([0., 0.], field_grid(self.settings)))
        self.field_ids = np.arange(-1, len(self.positions) - 1, dtype=np.int64)
        self.full_size = int(np.around(self.params['main']['pixel_pupil'] * self.settings['nd'] / 2) * 2)
        self.stamp_size = self.settings.get('stamp_size') or self.full_size
        self.groups = [(self.root / 'on_axis', np.array([-1], dtype=np.int64))]
        covered = set()
        for batch in sorted((self.root / 'field').glob('batch_*')):
            if not (batch / 'COMPLETE.json').is_file():
                raise ValueError(f'Incomplete field batch: {batch}')
            bounds = json.loads((batch / 'COMPLETE.json').read_text())
            first, stop = bounds['first'], bounds['stop']
            if (not isinstance(first, int) or not isinstance(stop, int)
                    or not 0 <= first < stop <= len(self.positions) - 1):
                raise ValueError(f'Invalid field range: {batch}')
            ids = np.arange(first, stop, dtype=np.int64)
            if covered.intersection(ids):
                raise ValueError('Overlapping field batches would duplicate PSF targets')
            covered.update(ids)
            self.groups.append((batch, ids))
        if covered != set(range(len(self.positions) - 1)):
            raise ValueError('PSF grid is incomplete; finish every field point before export')
        for directory, _ in self.groups:
            for band in ('J', 'K'):
                self._check_files(directory / band / 'short', 'exposure', '.fits')

    def _check_files(self, directory, prefix, suffix):
        expected = {f'{prefix}_{i:06d}{suffix}' for i in range(self.count)}
        if {p.name for p in directory.glob(f'{prefix}_*{suffix}')} != expected:
            raise ValueError(f'Missing or extra windows in {directory}')

    def _psfs(self, index, band):
        cube = None
        wavelength = self.settings['bands_nm'][band]
        pixel_scale = wavelength * 1e-9 / (self.full_size * self.params['main']['pixel_pitch'])
        pixel_scale *= 180 / np.pi * 3600 * 1000
        start = (self.start_step + index * 100) * self.dt
        crop = self.full_size // 2 - self.stamp_size // 2
        for directory, ids in self.groups:
            path = directory / band / 'short' / f'exposure_{index:06d}.fits'
            with fits.open(path, memmap=False) as hdul:
                header, data = hdul[0].header, hdul[0].data
                exact = {'BAND': band, 'NFRAMES': 100, 'NFIELDS': len(ids),
                         'NORM': 'FULLPSF', 'AXORDER': 'field,y,x',
                         'FULLNX': self.full_size, 'FULLNY': self.full_size,
                         'CROPX0': crop, 'CROPY0': crop}
                numeric = {'TSTART': start, 'TEND': start + 100 * self.dt,
                           'EXPTIME': 100 * self.dt, 'TIMEDEL': self.dt,
                           'WAVELEN': wavelength, 'PIXSCALE': pixel_scale}
                if (any(header.get(k) != v for k, v in exact.items())
                        or any(not np.isclose(header.get(k, np.nan), v, rtol=1e-10, atol=1e-10)
                               for k, v in numeric.items())):
                    raise ValueError(f'PSF exposure timing, sampling or band mismatch: {path}')
                coords = hdul['FIELD'].data
                if (not np.array_equal(coords['FIELD_ID'], ids)
                        or not np.allclose(np.column_stack((coords['X_ARCSEC'], coords['Y_ARCSEC'])),
                                           self.positions[ids + 1], rtol=0, atol=1e-10)):
                    raise ValueError(f'PSF field coordinates do not match the grid: {path}')
                if (data.shape != (len(ids), self.stamp_size, self.stamp_size)
                        or data.dtype.kind != 'f' or not np.isfinite(data).all()):
                    raise ValueError(f'Invalid PSF cube: {path}')
                dtype = data.dtype.newbyteorder('=')
                if cube is None:
                    cube = np.empty((len(self.positions), self.stamp_size, self.stamp_size), dtype=dtype)
                if cube.dtype != dtype:
                    raise ValueError('PSF precision differs between field batches')
                cube[ids + 1] = data
        return cube, pixel_scale

    def samples(self):
        # The saved controller stream is also the source of every field replay.
        # Compare it against the capture instead of joining files by index alone.
        with fits.open(self.root / 'comm.fits', memmap=False) as comm:
            if (comm[0].header.get('DOWNSAMP', 1) != 1
                    or not np.array_equal(comm[1].data, np.arange(self.total_steps) * self.step)
                    or comm[0].data.shape[0] != self.total_steps):
                raise ValueError('Replay commands are incomplete or downsampled')
            for index in range(self.count):
                with np.load(self.telemetry / f'window_{index:06d}.npz', allow_pickle=False) as saved:
                    arrays = {name: saved[name] for name in saved.files}
                first = self.start_step + index * 100
                ticks = np.arange(first, first + 100, dtype=np.int64) * self.step
                command_ticks = ticks - self.step
                command_ticks[ticks == 0] = -1
                if (not np.array_equal(arrays['time_ticks'], ticks)
                        or not np.array_equal(arrays['dm_command_time_ticks'], command_ticks)
                        or not np.array_equal(arrays['controller_commands'], comm[0].data[first:first + 100])):
                    raise ValueError(f'Telemetry/command timestamp or value mismatch in window {index}')
                pupil_size = self.params['main']['pixel_pupil']
                shapes = {
                    'dm_commands': (100, self.params['dm']['nmodes']),
                    'controller_commands': (100, self.params['dm']['nmodes']),
                    'shwfs_frames': (100, *self.params['detector_truth_sh']['size'][::-1]),
                    'residual_phase': (100, pupil_size, pupil_size),
                    'pupil': (pupil_size, pupil_size),
                    'time_ticks': (100,), 'dm_command_time_ticks': (100,),
                }
                if set(arrays) != set(shapes) or any(
                        arrays[k].shape != shape or not np.isfinite(arrays[k]).all()
                        for k, shape in shapes.items()):
                    raise ValueError(f'Invalid telemetry arrays in window {index}')
                scales = {}
                for band in ('J', 'K'):
                    arrays[f'psf_{band.lower()}'], scales[band] = self._psfs(index, band)
                arrays['field_ids'] = self.field_ids
                arrays['field_xy_arcsec'] = self.positions
                metadata = {
                    **self.meta, 'window_index': index,
                    'start_tick': int(ticks[0]), 'end_tick': int(ticks[-1] + self.step),
                    'first_simulation_step': first, 'exposure_seconds': 100 * self.dt,
                    'bands_nm': self.settings['bands_nm'], 'pixel_scale_mas': scales,
                    'grid_size': self.settings['grid_size'],
                    'field_width_arcsec': self.settings['field_width_arcsec'],
                    'psf_axes': ['field', 'y', 'x'], 'on_axis_index': 0,
                    'grid_order': 'remaining fields: row-major, x varies fastest',
                    'psf_normalization': 'unit flux before cropping; stamps not renormalized',
                    'full_psf_size': self.full_size, 'stamp_size': self.stamp_size,
                    'dm_sign': self.params['dm'].get('sign', -1),
                }
                yield {
                    '__key__': f'{self.run_id}_{index:06d}',
                    **{f'{k}.npy': npy_bytes(v) for k, v in arrays.items()},
                    'meta.json': json.dumps(metadata).encode('utf-8'),
                }


def export_dataset(run_dirs, output_dir, shard_size=4, max_shard_bytes=1024**3):
    try:
        import webdataset as wds
    except ImportError as exc:
        raise ImportError('Install dataset support with: python -m pip install -e ".[dataset]"') from exc
    if shard_size < 1 or max_shard_bytes < 1:
        raise ValueError('Shard size and byte limit must be positive')
    runs = [RunDataset(path) for path in run_dirs]
    if not runs or len({run.run_id for run in runs}) != len(runs):
        raise ValueError('Provide at least one run, with no duplicate run IDs')
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{output.name}.partial-', dir=output.parent))
    shards = []
    try:
        def completed_shard(filename):
            path = Path(filename)
            shards.append({'file': path.name, 'samples': writer.count, 'bytes': path.stat().st_size})

        with wds.ShardWriter(str(temporary / 'data-%06d.tar'), maxcount=shard_size,
                             maxsize=max_shard_bytes, encoder=False, post=completed_shard) as writer:
            for run in runs:
                provenance = temporary / 'runs' / run.run_id
                provenance.mkdir(parents=True)
                for name in ('params.yml', 'replay_params.yml', 'psf_grid.yml'):
                    shutil.copyfile(run.root / name, provenance / name)
                shutil.copyfile(run.telemetry / 'metadata.json', provenance / 'telemetry.json')
                for sample in run.samples():
                    writer.write(sample)
        manifest = {
            'schema_version': 1, 'samples': sum(run.count for run in runs), 'shards': shards,
            'runs': [{'run_id': run.run_id, 'source': str(run.root), 'samples': run.count} for run in runs],
        }
        (temporary / 'dataset.json').write_text(json.dumps(manifest, indent=2) + '\n')
        if output.exists():
            raise FileExistsError(output)
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dirs', type=Path, nargs='+', help='Complete runs captured with --dataset')
    parser.add_argument('--out', type=Path, required=True, help='New directory for data-XXXXXX.tar shards')
    parser.add_argument('--shard-size', type=int, default=4, help='Maximum samples per shard (default: 4)')
    parser.add_argument('--max-shard-gib', type=float, default=1.,
                        help='Rotate after this size; one sample may exceed it (default: 1 GiB)')
    args = parser.parse_args()
    if not np.isfinite(args.max_shard_gib) or args.max_shard_gib <= 0:
        parser.error('--max-shard-gib must be finite and positive')
    manifest = export_dataset(args.run_dirs, args.out, args.shard_size, int(args.max_shard_gib * 1024**3))
    print(f"Saved {manifest['samples']} samples in {len(manifest['shards'])} shards: {args.out}")


if __name__ == '__main__':
    main()
