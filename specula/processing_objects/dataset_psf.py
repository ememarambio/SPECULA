"""Optional PSF implementation for dataset generation."""

import logging

from specula.lib.calc_psf import calc_psf
from specula.processing_objects.psf import PSF


class DatasetPSF(PSF):
    """PSF with lazy diagnostic logging, preserving the standard PSF outputs.

    Unlike eager scalar formatting, disabled INFO logging never reads a GPU
    Strehl value back to the host. The original PSF class remains available.
    """

    def trigger_code(self):
        ef = self.local_inputs['in_ef']
        self.psf.value, self.total_psf = calc_psf(
            ef.phi_at_lambda(self.wavelengthInNm), ef.A,
            imwidth=self.out_size[0], normalize=True, xp=self.xp,
            complex_dtype=self.complex_dtype, return_total=True)
        center = tuple(size // 2 for size in self.out_size)
        self.sr.value = self.psf.value[center] / self.ref.i[center]
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info('SR at %dnm : %s', int(self.wavelengthInNm), self.sr.value)
