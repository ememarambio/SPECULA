"""Window averages of normalized PSF stamps at their native sampling."""

from iteras.processing_objects.windowed_integration import WindowedIntegration


class PsfStampIntegration(WindowedIntegration):
    """Crop a full PSF before averaging; publish only complete exposures.

    The input must already be normalized over the full image. Cropping never
    renormalizes its flux. Outputs retain the previous completed exposure until
    the next boundary; generation_time identifies the exposure's last frame.
    """

    def __init__(self, simul_params, full_size: int, stamp_size: int, dt: float,
                 start_time: float = 0., target_device_idx: int = None,
                 precision: int = None):
        if (not isinstance(full_size, int) or not isinstance(stamp_size, int)
                or not 1 <= stamp_size <= full_size):
            raise ValueError('stamp_size must fit within full_size')
        super().__init__(simul_params, [stamp_size, stamp_size], dt,
                         start_time=start_time, update_time_on_dt=True,
                         target_device_idx=target_device_idx, precision=precision)
        self.full_size = full_size
        crop = full_size // 2 - stamp_size // 2
        self.stamp_slice = (slice(crop, crop + stamp_size),) * 2
        self.frame_count = 0

    def trigger_code(self):
        if self.current_time < self.start_time:
            return
        value = self.local_inputs['input'].value
        if value.shape != (self.full_size, self.full_size):
            raise ValueError('Expected a full-size normalized PSF')
        self.integrated_value += value[self.stamp_slice]
        self.frame_count += 1
        if (self.current_time + self.loop_dt - self.start_time) % self.dt == 0:
            if self.frame_count != self.dt // self.loop_dt:
                raise ValueError('Incomplete PSF exposure')
            self.output.value[:] = self.integrated_value / self.frame_count
            self.integrated_value.fill(0)
            self.frame_count = 0
            self.output.generation_time = self.current_time
