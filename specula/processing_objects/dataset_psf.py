"""Optional PSF implementation for dataset generation."""

import logging
import numpy as np

from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.simul_params import SimulParams
from specula.lib.calc_psf import calc_psf
from specula.processing_objects.psf import PSF


class DatasetPSF(PSF):
    """PSF with lazy diagnostic logging, preserving the standard PSF outputs.

    Unlike eager scalar formatting, disabled INFO logging never reads a GPU
    Strehl value back to the host. The original PSF class remains available.
    With compute_statistics=False only out_psf is produced: no Strehl reference,
    cumulative PSF, variance or profile buffers are allocated. The default keeps
    the full standard output set.
    """

    def __init__(self, simul_params: SimulParams, wavelengthInNm: float,
                 nd: float = None, pixel_size_mas: float = None,
                 start_time: float = 0., compute_profile_metrics: bool = False,
                 compute_metrics_in_trigger: bool = False,
                 ee_radius_in_lambda_d=None, target_device_idx: int = None,
                 precision: int = None, compute_statistics: bool = True):
        if not compute_statistics and compute_profile_metrics:
            raise ValueError('Profile metrics require compute_statistics=True')
        super().__init__(simul_params, wavelengthInNm, nd, pixel_size_mas,
                         start_time, compute_profile_metrics, compute_metrics_in_trigger,
                         ee_radius_in_lambda_d, target_device_idx, precision)
        self.compute_statistics = compute_statistics

    def setup(self):
        if self.compute_statistics:
            return super().setup()
        BaseProcessingObj.setup(self)
        self.out_size = [int(np.around(dim * self.nd / 2) * 2)
                         for dim in self.local_inputs['in_ef'].size]

    def prepare_trigger(self, t):
        if self.compute_statistics:
            return super().prepare_trigger(t)
        BaseProcessingObj.prepare_trigger(self, t)

    def trigger_code(self):
        ef = self.local_inputs['in_ef']
        self.psf.value, self.total_psf = calc_psf(
            ef.phi_at_lambda(self.wavelengthInNm), ef.A,
            imwidth=self.out_size[0], normalize=True, xp=self.xp,
            complex_dtype=self.complex_dtype, return_total=True)
        if self.compute_statistics:
            center = tuple(size // 2 for size in self.out_size)
            self.sr.value = self.psf.value[center] / self.ref.i[center]
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info('SR at %dnm : %s', int(self.wavelengthInNm), self.sr.value)

    def post_trigger(self):
        if self.compute_statistics:
            return super().post_trigger()
        BaseProcessingObj.post_trigger(self)
        self.psf.generation_time = self.current_time

    def finalize(self):
        if self.compute_statistics:
            super().finalize()
