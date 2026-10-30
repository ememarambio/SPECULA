"""Stream completed, synchronized PSF exposure windows to FITS."""

from pathlib import Path

import numpy as np
from astropy.io import fits

from specula import cpuArray
from specula.base_processing_obj import BaseProcessingObj, InputDesc
from specula.base_value import BaseValue
from specula.connections import InputList
from specula.data_objects.simul_params import SimulParams


class PsfExposureStore(BaseProcessingObj):
    """Write one [field, y, x] cube per completed exposure, without buffering time.

    Connect ``in_psfs`` to WindowedIntegration outputs with
    ``update_time_on_dt=True`` and identical start times/durations. ``positions``
    contains [x, y] sky offsets in arcseconds, in the same order as the inputs.
    ``field_ids`` identifies those positions in the full science grid.
    Exposure bounds are [TSTART, TEND), in simulation seconds; the producer's
    generation time labels the *last sample*, one time_step before TEND.
    Incomplete windows are never written. Existing files are never overwritten.
    """

    def __init__(self, simul_params: SimulParams, store_dir: str, band: str,
                 wavelength_nm: float, pixel_size_mas: float, n_frames: int,
                 positions: list, field_ids: list, start_time: float = 0):
        # Disk I/O runs on CPU; InputList transfers only completed exposures.
        super().__init__(target_device_idx=-1)
        if int(n_frames) != n_frames or n_frames < 1:
            raise ValueError('n_frames must be a positive integer')
        self.loop_dt = self.seconds_to_t(simul_params.time_step)
        self.start = self.seconds_to_t(start_time)
        if self.loop_dt <= 0 or self.start < 0 or self.start % self.loop_dt:
            raise ValueError('start_time must be nonnegative and aligned to time_step')
        self.n_frames = int(n_frames)
        self.duration = self.loop_dt * self.n_frames
        self.positions = np.asarray(positions, dtype=float)
        self.field_ids = np.asarray(field_ids, dtype=np.int64)
        if (self.positions.shape != (len(self.field_ids), 2)
                or not len(self.field_ids) or not np.isfinite(self.positions).all()
                or len(set(self.field_ids)) != len(self.field_ids)):
            raise ValueError('Expected unique field_ids and matching finite [x, y] positions')
        if wavelength_nm <= 0 or pixel_size_mas <= 0:
            raise ValueError('Wavelength and pixel size must be positive')
        self.store_dir = Path(store_dir)
        self.band = band
        self.wavelength_nm = wavelength_nm
        self.pixel_size_mas = pixel_size_mas
        self.last_written = -1
        self.inputs['in_psfs'] = InputList(type=BaseValue)

    @classmethod
    def input_names(cls):
        return {'in_psfs': InputDesc(BaseValue, 'Completed PSFs, one per field position')}

    @classmethod
    def output_names(cls):
        return {}

    def setup(self):
        super().setup()
        if len(self.local_inputs['in_psfs']) != len(self.field_ids):
            raise ValueError('One PSF input is required per field position')
        self.store_dir.mkdir(parents=True, exist_ok=True)
        if any(self.store_dir.glob('exposure_*.fits')):
            raise FileExistsError(f'Exposure output already exists in {self.store_dir}')

    def trigger_code(self):
        if self.current_time <= self.last_written:
            return
        psfs = self.local_inputs['in_psfs']
        if any(p.generation_time != self.current_time for p in psfs):
            raise ValueError('PSF exposure inputs must update together')
        end = self.current_time + self.loop_dt
        elapsed = end - self.start
        if elapsed <= 0 or elapsed % self.duration:
            raise ValueError('PSF input arrived outside a complete exposure boundary')
        index = elapsed // self.duration - 1
        cube = np.stack([cpuArray(p.value) for p in psfs])
        if cube.ndim != 3 or not np.isfinite(cube).all():
            raise ValueError('Expected finite 2D PSFs')
        header = fits.Header()
        header.update({
            'BAND': self.band, 'WAVELEN': (self.wavelength_nm, 'Wavelength [nm]'),
            'PIXSCALE': (self.pixel_size_mas, 'PSF pixel scale [mas/pixel]'),
            'NFRAMES': self.n_frames, 'NFIELDS': len(psfs),
            'TSTART': (self.t_to_seconds(end - self.duration), 'Exposure start [s], inclusive'),
            'TEND': (self.t_to_seconds(end), 'Exposure end [s], exclusive'),
            'EXPTIME': (self.t_to_seconds(self.duration), 'Exposure duration [s]'),
            'TIMEDEL': (self.t_to_seconds(self.loop_dt), 'Simulation step [s]'),
            'BUNIT': 'normalized intensity', 'AXORDER': 'field,y,x',
        })
        coords = fits.BinTableHDU.from_columns([
            fits.Column(name='FIELD_ID', format='K', array=self.field_ids),
            fits.Column(name='X_ARCSEC', format='D', unit='arcsec', array=self.positions[:, 0]),
            fits.Column(name='Y_ARCSEC', format='D', unit='arcsec', array=self.positions[:, 1]),
        ], name='FIELD')
        path = self.store_dir / f'exposure_{index:06d}.fits'
        # A crash cannot leave a partly written file looking like a finished cube.
        temporary = path.with_suffix('.fits.partial')
        if path.exists():
            raise FileExistsError(path)
        fits.HDUList([fits.PrimaryHDU(cube, header), coords]).writeto(temporary)
        temporary.rename(path)
        self.last_written = self.current_time
