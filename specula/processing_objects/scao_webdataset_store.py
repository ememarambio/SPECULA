"""Write synchronized SCAO samples directly to WebDataset during simulation."""

import io
import json
import uuid
from pathlib import Path

import numpy as np
import yaml

from specula import cpuArray
from specula.base_processing_obj import BaseProcessingObj, InputDesc
from specula.base_value import BaseValue
from specula.connections import InputList, InputValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.pixels import Pixels
from specula.data_objects.simul_params import SimulParams
from specula.lib.calc_psf import calc_psf_geometry


def npy_bytes(array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


class _ShardStream:
    """Publish closed shards; leave a failing/open shard marked as partial."""

    def __init__(self, root, maxcount, maxsize):
        import webdataset as wds

        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.shards = []
        self.failed = False
        self.closed = False
        self.writer = wds.ShardWriter(str(self.root / 'data-%06d.tar.partial'),
                                      maxcount=maxcount, maxsize=maxsize,
                                      encoder=False, post=self._completed, verbose=0)

    def _completed(self, filename):
        if self.failed:
            return
        path = Path(filename)
        finished = path.with_suffix('')
        if finished.exists():
            raise FileExistsError(finished)
        path.rename(finished)
        self.shards.append({'file': finished.name, 'samples': self.writer.count,
                            'bytes': finished.stat().st_size})

    def write(self, sample):
        try:
            self.writer.write(sample)
        except BaseException:
            self.failed = True
            raise

    def close(self, abort=False):
        self.failed |= abort
        if not self.closed:
            self.closed = True
            self.writer.close()


class ScaoWebdatasetStore(BaseProcessingObj):
    """One 100-step telemetry sample with full-field J/K PSFs, written in place.

    PSF lists come from independent WindowedIntegration objects using identical
    start times and update_time_on_dt=True. The field axis is on-axis first,
    then the entire square grid in row-major order. Stay on the producers'
    device: readiness checks must not clone full PSF buffers to the host.
    Only completed central stamps are transferred. Long exposures go to a
    separate stream, without buffering their full telemetry sequences.
    """

    def __init__(self, simul_params: SimulParams, store_dir: str, grid_size: int,
                 field_width_arcsec: float, bands_nm: dict, nd: float = 2.,
                 stamp_size: int = 128, n_frames: int = 100,
                 long_frames: int = 5000, start_time: float = .2,
                 shard_size: int = 4, max_shard_bytes: int = 1073741824,
                 dm_sign: int = -1, target_device_idx: int = None,
                 psfs_are_stamps: bool = False, device_buffering: bool = False):
        super().__init__(target_device_idx=target_device_idx)
        if set(bands_nm) != {'J', 'K'} or any(not np.isfinite(v) or v <= 0 for v in bands_nm.values()):
            raise ValueError('Expected positive J/K wavelengths')
        phase_wavelength_nm = bands_nm['J']
        self.step = self.seconds_to_t(simul_params.time_step)
        self.start = self.seconds_to_t(start_time)
        self.end = self.seconds_to_t(simul_params.total_time)
        if (not isinstance(n_frames, int) or n_frames < 1 or self.step <= 0
                or self.start < 0 or self.start % self.step
                or self.end <= self.start or (self.end - self.start) % (n_frames * self.step)):
            raise ValueError('Telemetry must contain complete windows aligned to time_step')
        if not np.isfinite(phase_wavelength_nm) or phase_wavelength_nm <= 0:
            raise ValueError('phase_wavelength_nm must be finite and positive')
        self.n_frames = n_frames
        self.wavelength = float(phase_wavelength_nm)
        self.store_dir = Path(store_dir)
        self.next_time = self.start
        self.frame = self.windows = 0
        self.buffers = None
        self.pupil = None
        self.device_buffering = device_buffering
        self._device_buffers = None
        self._device_pupil = None
        self._pupil_changed = None
        self.inputs['in_commands'] = InputValue(type=BaseValue)
        self.inputs['in_applied_commands'] = InputValue(type=BaseValue)
        self.inputs['in_pixels'] = InputValue(type=Pixels)
        self.inputs['in_ef'] = InputValue(type=ElectricField)

        if n_frames != 100:
            raise ValueError('WebDataset samples require exactly 100 frames')
        if (not isinstance(long_frames, int) or long_frames < 1
                or (self.end - self.start) % (long_frames * self.step)):
            raise ValueError('Science duration must contain complete long exposures')
        if (not isinstance(grid_size, int) or grid_size < 2
                or not np.isfinite(field_width_arcsec) or field_width_arcsec <= 0):
            raise ValueError('Expected a finite square field and grid_size >= 2')
        if (not isinstance(shard_size, int) or shard_size < 1
                or not isinstance(max_shard_bytes, int) or max_shard_bytes < 1):
            raise ValueError('Shard limits must be positive integers')
        self.long_frames = long_frames
        self.long_windows = 0
        self.shard_size, self.max_shard_bytes = shard_size, max_shard_bytes
        self.grid_size, self.field_width = grid_size, field_width_arcsec
        axis = np.linspace(-field_width_arcsec / 2, field_width_arcsec / 2, grid_size)
        x, y = np.meshgrid(axis, axis, indexing='xy')
        self.positions = np.vstack(([0., 0.], np.column_stack((x.ravel(), y.ravel()))))
        self.field_ids = np.arange(-1, grid_size**2, dtype=np.int64)
        self.bands = dict(bands_nm)
        if not np.isfinite(nd) or nd < 1:
            raise ValueError('nd must be finite and at least 1')
        self.full_size = int(np.around(simul_params.pixel_pupil * nd / 2) * 2)
        self.stamp_size = self.full_size if stamp_size is None else stamp_size
        if not isinstance(self.stamp_size, int) or not 1 <= self.stamp_size <= self.full_size:
            raise ValueError('stamp_size must fit within the full PSF')
        self.crop = self.full_size // 2 - self.stamp_size // 2
        self.stamp_slice = (slice(self.crop, self.crop + self.stamp_size),) * 2
        self.psfs_are_stamps = psfs_are_stamps
        self.scales = {band: calc_psf_geometry(simul_params.pixel_pupil, simul_params.pixel_pitch,
                                              wavelength, nd=nd).pixel_size_mas
                       for band, wavelength in self.bands.items()}
        self.dm_sign = dm_sign
        self.streams = {}
        self.simulation_params = None
        self.resolved_seeds = {}
        for label in ('short', 'long'):
            for band in ('J', 'K'):
                self.inputs[f'in_{label}_{band}'] = InputList(type=BaseValue)

    @classmethod
    def input_names(cls):
        return {
            'in_commands': InputDesc(BaseValue, 'Controller commands emitted at this step'),
            'in_applied_commands': InputDesc(BaseValue, 'DM commands applied to this exposure (:-1)'),
            'in_pixels': InputDesc(Pixels, 'Same-step SHWFS detector frame'),
            'in_ef': InputDesc(ElectricField, 'Same-step on-axis residual electric field'),
            **{f'in_{label}_{band}': InputDesc(BaseValue, f'Complete {label} {band} exposures, on-axis then grid')
               for label in ('short', 'long') for band in ('J', 'K')},
        }

    @classmethod
    def output_names(cls):
        return {}

    def set_simulation_params(self, params):
        self.simulation_params = params

    def set_replay_params(self, replay_params):
        self.resolved_seeds = (replay_params.get('data_source') or {}).get('random_seeds', {})

    def setup(self):
        # Check the original references before BaseProcessingObj.get_all_inputs
        # can create cross-device clones during setup or readiness checks.
        for connection in self.inputs.values():
            for item in connection.input_values:
                if item.output_ref is not None and item.output_ref.target_device_idx != self.target_device_idx:
                    raise ValueError('Keep WebDataset and its producers on the same device')
        BaseProcessingObj.setup(self)
        self._setup_telemetry()
        for label in ('short', 'long'):
            for band in ('J', 'K'):
                psfs = self.local_inputs[f'in_{label}_{band}']
                if (len(psfs) != len(self.field_ids)
                        or any(p.value.shape != self._psf_input_shape() for p in psfs)):
                    raise ValueError('Expected one correctly sized PSF per field point, including on-axis')
        self.store_dir.mkdir(parents=True, exist_ok=False)
        self.metadata.update({
            'global_precision': int(self.precision), 'random_seeds': self.resolved_seeds,
            'bands_nm': self.bands, 'pixel_scale_mas': self.scales,
            'grid_size': self.grid_size, 'field_width_arcsec': self.field_width,
            'psf_axes': ['field', 'y', 'x'], 'on_axis_index': 0,
            'grid_order': 'remaining fields: row-major, x varies fastest',
            'psf_normalization': 'unit flux before cropping; stamps not renormalized',
            'full_psf_size': self.full_size, 'stamp_size': self.stamp_size,
            'crop_x0': self.crop, 'crop_y0': self.crop, 'dm_sign': self.dm_sign,
        })
        (self.store_dir / 'metadata.json').write_text(json.dumps(self.metadata, indent=2) + '\n')
        if self.simulation_params is not None:
            (self.store_dir / 'params.yml').write_text(yaml.safe_dump(self.simulation_params, sort_keys=False))
        try:
            self.streams['short'] = _ShardStream(self.store_dir, self.shard_size, self.max_shard_bytes)
            self.streams['long'] = _ShardStream(self.store_dir / 'long_exposures',
                                               self.shard_size, self.max_shard_bytes)
        except BaseException:
            self.abort()
            raise

    def _setup_telemetry(self):
        """Initialize window metadata independently of the output format."""
        self.metadata = {
            'schema_version': 1, 'run_id': uuid.uuid4().hex,
            'n_frames': self.n_frames, 'start_tick': self.start,
            'end_tick': self.end, 'step_ticks': self.step,
            'tick_seconds': self.t_to_seconds(1),
            'phase_wavelength_nm': self.wavelength, 'phase_unit': 'rad',
            'phase_direction_arcsec': [0., 0.],
            'phase_convention': 'unwrapped; zero outside pupil; piston retained',
            'dm_commands': 'applied during this exposure, before DM sign',
            'controller_commands': 'emitted at the same simulation step',
            'applied_command_delay_steps': 1,
        }

    def _validate_step(self):
        if self.current_time != self.next_time or self.current_time >= self.end:
            raise ValueError('Missing, repeated or out-of-order telemetry step')
        for name in ('in_commands', 'in_pixels', 'in_ef'):
            if self.local_inputs[name].generation_time != self.current_time:
                raise ValueError(f'{name} is not from the current simulation step')
        applied = self.local_inputs['in_applied_commands']
        # At t=0 the untriggered DM represents its initial zero command.
        expected = self.current_time - self.step if self.current_time else -1
        if applied.generation_time != expected:
            raise ValueError('Applied DM command must be read before the current DM update')

    def _capture_step(self):
        if self.device_buffering:
            return self._capture_device_step()
        if self.current_time < self.start:
            return
        self._validate_step()
        applied = self.local_inputs['in_applied_commands']

        ef = self.local_inputs['in_ef']
        pupil = cpuArray(ef.A)
        if self.pupil is None:
            self.pupil = pupil.copy()
        elif not np.array_equal(pupil, self.pupil):
            raise ValueError('Dataset capture requires a static on-axis pupil')
        phase = cpuArray(ef.phaseInNm, force_copy=True)
        phase *= 2 * np.pi / self.wavelength
        phase[self.pupil == 0] = 0
        values = {
            'dm_commands': cpuArray(applied.value),
            'controller_commands': cpuArray(self.local_inputs['in_commands'].value),
            'shwfs_frames': cpuArray(self.local_inputs['in_pixels'].pixels),
            'residual_phase': phase,
        }
        if self.buffers is None:
            self.buffers = {k: np.empty((self.n_frames, *v.shape), dtype=v.dtype)
                            for k, v in values.items()}
            self.buffers['time_ticks'] = np.empty(self.n_frames, dtype=np.int64)
            self.buffers['dm_command_time_ticks'] = np.empty(self.n_frames, dtype=np.int64)
        for name, value in values.items():
            if value.shape != self.buffers[name].shape[1:] or not np.isfinite(value).all():
                raise ValueError(f'Invalid or changing {name} array')
            self.buffers[name][self.frame] = value
        self.buffers['time_ticks'][self.frame] = self.current_time
        self.buffers['dm_command_time_ticks'][self.frame] = applied.generation_time
        self.frame += 1
        self.next_time += self.step
        if self.frame == self.n_frames:
            self._write_window()
            self.windows += 1
            self.frame = 0

    def _capture_device_step(self):
        """Snapshot matching frames on-device; transfer only complete windows.

        Timestamp and shape checks run every step. Static-pupil and finite-value
        checks are resolved before publishing each completed window, avoiding
        host synchronization in the inner loop.
        """
        if self.current_time < self.start:
            return
        self._validate_step()
        ef = self.local_inputs['in_ef']
        applied = self.local_inputs['in_applied_commands']
        values = {
            'dm_commands': applied.value,
            'controller_commands': self.local_inputs['in_commands'].value,
            'shwfs_frames': self.local_inputs['in_pixels'].pixels,
            'residual_phase': ef.phaseInNm,
        }
        if self._device_buffers is None:
            self._device_buffers = {
                name: self.xp.empty((self.n_frames, *value.shape), dtype=value.dtype)
                for name, value in values.items()}
            self._device_pupil = ef.A.copy()
            self._pupil_changed = self.xp.asarray(False)
            self.pupil = cpuArray(ef.A, force_copy=True)
            self.buffers = {name: np.empty(self.n_frames, dtype=np.int64)
                            for name in ('time_ticks', 'dm_command_time_ticks')}
        if ef.A.shape != self._device_pupil.shape:
            raise ValueError('Dataset capture requires a static on-axis pupil')
        self._pupil_changed |= self.xp.any(ef.A != self._device_pupil)
        for name, value in values.items():
            if value.shape != self._device_buffers[name].shape[1:]:
                raise ValueError(f'Invalid or changing {name} array')
            self._device_buffers[name][self.frame] = value
        self.buffers['time_ticks'][self.frame] = self.current_time
        self.buffers['dm_command_time_ticks'][self.frame] = applied.generation_time
        self.frame += 1
        self.next_time += self.step
        if self.frame == self.n_frames:
            if bool(cpuArray(self._pupil_changed)):
                raise ValueError('Dataset capture requires a static on-axis pupil')
            for name, value in self._device_buffers.items():
                host = cpuArray(value, force_copy=True)
                if name == 'residual_phase':
                    host *= 2 * np.pi / self.wavelength
                    host[:, self.pupil == 0] = 0
                if not np.isfinite(host).all():
                    raise ValueError(f'Invalid or changing {name} array')
                self.buffers[name] = host
            self._write_window()
            self.windows += 1
            self.frame = 0

    def _validate_complete(self):
        if self.next_time != self.end or self.frame:
            raise ValueError('Incomplete telemetry: not all requested simulation steps were captured')

    def _psf_members(self, label):
        members = {}
        for band in ('J', 'K'):
            psfs = self.local_inputs[f'in_{label}_{band}']
            if any(p.generation_time != self.current_time for p in psfs):
                raise ValueError(f'{label} {band} PSFs do not match the completed telemetry exposure')
            if self.device_buffering:
                cube = cpuArray(self.xp.stack([self._psf_stamp(p) for p in psfs]))
            else:
                cube = np.stack([cpuArray(self._psf_stamp(p)) for p in psfs])
            if not np.isfinite(cube).all():
                raise ValueError('PSF targets must be finite')
            members[f'psf_{band.lower()}.npy'] = npy_bytes(cube)
        members['field_ids.npy'] = npy_bytes(self.field_ids)
        members['field_xy_arcsec.npy'] = npy_bytes(self.positions)
        return members

    def _psf_input_shape(self):
        size = self.stamp_size if self.psfs_are_stamps else self.full_size
        return (size, size)

    def _psf_stamp(self, psf):
        return psf.value if self.psfs_are_stamps else psf.value[self.stamp_slice]

    def _sample_metadata(self, frames, index, kind):
        end = self.current_time + self.step
        return {**self.metadata, 'sample_kind': kind, 'window_index': index,
                'n_frames': frames, 'start_tick': end - frames * self.step, 'end_tick': end,
                'first_simulation_step': end // self.step - frames,
                'exposure_seconds': self.t_to_seconds(frames * self.step)}

    def _write_window(self):
        sample = self._psf_members('short')
        sample.update({f'{name}.npy': npy_bytes(value) for name, value in self.buffers.items()})
        sample['pupil.npy'] = npy_bytes(self.pupil)
        sample['__key__'] = f"{self.metadata['run_id']}_{self.windows:06d}"
        sample['meta.json'] = json.dumps(self._sample_metadata(100, self.windows, 'training')).encode()
        self.streams['short'].write(sample)

    def trigger_code(self):
        self._capture_step()
        if self.current_time < self.start:
            return
        if (self.current_time + self.step - self.start) % (self.long_frames * self.step) == 0:
            sample = self._psf_members('long')
            sample['__key__'] = f"{self.metadata['run_id']}_long_{self.long_windows:06d}"
            sample['meta.json'] = json.dumps(
                self._sample_metadata(self.long_frames, self.long_windows, 'long_psf')).encode()
            self.streams['long'].write(sample)
            self.long_windows += 1

    def finalize(self):
        try:
            self._validate_complete()
            if self.long_windows * self.long_frames * self.step != self.end - self.start:
                raise ValueError('Incomplete long PSF exposure stream')
            for stream in self.streams.values():
                stream.close()
            for label, count in [('long', self.long_windows), ('short', self.windows)]:
                stream = self.streams[label]
                manifest = {'schema_version': 1, 'run_id': self.metadata['run_id'],
                            'samples': count, 'shards': stream.shards}
                if label == 'short':
                    manifest['long_exposures'] = 'long_exposures/dataset.json'
                path = stream.root / 'dataset.json'
                temporary = path.with_suffix('.json.partial')
                temporary.write_text(json.dumps(manifest, indent=2) + '\n')
                temporary.rename(path)
        except BaseException:
            self.abort()
            raise

    def abort(self):
        """Close open handles without publishing partial shards or a manifest."""
        for stream in self.streams.values():
            try:
                stream.close(abort=True)
            except Exception:
                self.logger.exception('Could not close an interrupted dataset shard')
