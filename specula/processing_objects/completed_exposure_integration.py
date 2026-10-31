"""Combine consecutive, equal-duration completed intensity exposures."""

from iteras.processing_objects.windowed_integration import WindowedIntegration


class CompletedExposureIntegration(WindowedIntegration):
    """Average non-overlapping input exposures, checking every boundary.

    input_frames and output_frames are counts of simulation steps. The output
    duration must be a multiple of the input duration; inputs must publish only
    at completed exposure boundaries, with the last frame's timestamp.
    """

    def __init__(self, simul_params, n_elem, input_frames: int, output_frames: int,
                 start_time: float = 0., target_device_idx: int = None,
                 precision: int = None):
        if (not isinstance(input_frames, int) or not isinstance(output_frames, int)
                or input_frames < 1 or output_frames < input_frames
                or output_frames % input_frames):
            raise ValueError('output_frames must be a positive multiple of input_frames')
        super().__init__(simul_params, n_elem, output_frames * simul_params.time_step,
                         start_time=start_time, update_time_on_dt=True,
                         target_device_idx=target_device_idx, precision=precision)
        self.input_frames = input_frames
        self.exposures_per_output = output_frames // input_frames
        self.next_input_time = self.start_time + (input_frames - 1) * self.loop_dt
        self.exposure_count = 0

    def trigger_code(self):
        value = self.local_inputs['input']
        if self.current_time < self.start_time:
            return
        if (self.current_time != self.next_input_time
                or value.generation_time != self.current_time):
            raise ValueError('Missing or out-of-order completed exposure')
        if value.value.shape != self.integrated_value.shape:
            raise ValueError('Completed exposure shape changed')
        self.integrated_value += value.value
        self.exposure_count += 1
        self.next_input_time += self.input_frames * self.loop_dt
        if self.exposure_count == self.exposures_per_output:
            self.output.value[:] = self.integrated_value / self.exposures_per_output
            self.integrated_value.fill(0)
            self.exposure_count = 0
            self.output.generation_time = self.current_time
