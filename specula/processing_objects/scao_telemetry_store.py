"""Save aligned SCAO training telemetry in bounded, non-overlapping windows."""

import json
from pathlib import Path
import uuid

import numpy as np

from specula import cpuArray
from specula.base_processing_obj import BaseProcessingObj, InputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.pixels import Pixels
from specula.data_objects.simul_params import SimulParams


class ScaoTelemetryStore(BaseProcessingObj):
    """Capture the commands, detector frames and on-axis phase for one window.

    Connect ``in_applied_commands`` to ``dm.out_clipped_command:-1``: the
    propagation also reads ``dm.out_layer:-1``. This makes the scheduler run
    this sink BEFORE the DM overwrites its previous command, even though the
    current controller output is already available. The controller's internal
    delay is already included in its output; do not shift it a second time.

    Phase is unwrapped radians at ``phase_wavelength_nm``, zero outside the
    pupil, without piston subtraction. Only a static pupil is supported.
    All buffers are on the host and contain at most ``n_frames`` samples.
    The object itself follows the simulation device to avoid input cloning.
    """

    def __init__(self, simul_params: SimulParams, store_dir: str,
                 n_frames: int = 100, start_time: float = 0.2,
                 phase_wavelength_nm: float = 1250.,
                 target_device_idx: int = None):
        super().__init__(target_device_idx=target_device_idx)
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
        self.inputs['in_commands'] = InputValue(type=BaseValue)
        self.inputs['in_applied_commands'] = InputValue(type=BaseValue)
        self.inputs['in_pixels'] = InputValue(type=Pixels)
        self.inputs['in_ef'] = InputValue(type=ElectricField)

    @classmethod
    def input_names(cls):
        return {
            'in_commands': InputDesc(BaseValue, 'Controller commands emitted at this step'),
            'in_applied_commands': InputDesc(BaseValue, 'DM commands applied to this exposure (:-1)'),
            'in_pixels': InputDesc(Pixels, 'Same-step SHWFS detector frame'),
            'in_ef': InputDesc(ElectricField, 'Same-step on-axis residual electric field'),
        }

    @classmethod
    def output_names(cls):
        return {}

    def setup(self):
        super().setup()
        self.store_dir.mkdir(parents=True, exist_ok=False)
        self._setup_telemetry()
        (self.store_dir / 'metadata.json').write_text(json.dumps(self.metadata, indent=2) + '\n')

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

    def trigger_code(self):
        if self.current_time < self.start:
            return
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

    def _write_window(self):
        path = self.store_dir / f'window_{self.windows:06d}.npz'
        temporary = path.with_suffix('.npz.partial')
        if path.exists():
            raise FileExistsError(path)
        with temporary.open('xb') as stream:
            np.savez(stream, **self.buffers, pupil=self.pupil)
        temporary.rename(path)

    def _validate_complete(self):
        if self.next_time != self.end or self.frame:
            raise ValueError('Incomplete telemetry: not all requested simulation steps were captured')

    def finalize(self):
        self._validate_complete()
        (self.store_dir / 'COMPLETE.json').write_text(
            json.dumps({'run_id': self.metadata['run_id'], 'windows': self.windows}) + '\n')
