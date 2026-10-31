"""Numerical and lifecycle checks for the optional dataset fast paths."""

import logging
from unittest.mock import Mock

import numpy as np
import pytest

import specula
if specula.global_precision is None:
    specula.init(-1)

from specula import cpuArray
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.dataset_psf import DatasetPSF
from specula.processing_objects.psf import PSF


DEVICES = [-1, pytest.param(0, marks=pytest.mark.skipif(
    specula.cp is None, reason='CuPy/GPU unavailable'))]


@pytest.mark.parametrize('device', DEVICES)
def test_dataset_psf_matches_standard_outputs(device):
    xp = np if device < 0 else specula.cp
    params = SimulParams(pixel_pupil=8, pixel_pitch=.1, time_step=.002)
    ef = ElectricField(8, 8, .1, target_device_idx=device)
    standard, fast = [cls(params, 1250., nd=2., target_device_idx=device)
                      for cls in (PSF, DatasetPSF)]
    for obj in (standard, fast):
        obj.inputs['in_ef'].set(ef)
        obj.setup()
    for step in range(4):
        ef.phaseInNm[:] = xp.arange(64).reshape(8, 8) * (step + 1)
        ef.generation_time = step * fast.seconds_to_t(.002)
        for obj in (standard, fast):
            obj.check_ready(ef.generation_time)
            obj.trigger()
            obj.post_trigger()
    for obj in (standard, fast):
        obj.finalize()
    for name in ('psf', 'sr', 'int_psf', 'int_sr', 'std_psf'):
        np.testing.assert_allclose(cpuArray(getattr(fast, name).value),
                                   cpuArray(getattr(standard, name).value))


def test_disabled_psf_logging_never_formats_device_scalar(monkeypatch):
    import specula.processing_objects.dataset_psf as module

    class DeviceScalar(np.ndarray):
        def __format__(self, spec):
            raise AssertionError('Device scalar copied to host')

        def __str__(self):
            raise AssertionError('Device scalar copied to host')

    obj = DatasetPSF(SimulParams(pixel_pupil=8, pixel_pitch=.1), 1250., nd=2.)
    ef = ElectricField(8, 8, .1, target_device_idx=-1)
    obj.inputs['in_ef'].set(ef)
    obj.setup()
    obj.ref.i[:] = 1
    # Indexing a subclass returns a NumPy scalar, so return a device-like 0D
    # array explicitly, as CuPy does for scalar indexing.
    class DeviceImage:
        def __getitem__(self, key):
            return np.asarray(.5).view(DeviceScalar)

    monkeypatch.setattr(module, 'calc_psf', lambda *a, **k: (DeviceImage(), 1.))
    obj.logger = Mock()
    obj.logger.isEnabledFor.return_value = False
    obj.trigger_code()
    obj.logger.isEnabledFor.assert_called_once_with(logging.INFO)
    obj.logger.info.assert_not_called()


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('stamp', [3, 4, 8])
def test_stamp_averages_match_full_psf_crop(device, stamp):
    from specula.base_value import BaseValue
    from iteras.processing_objects.windowed_integration import WindowedIntegration
    from specula.processing_objects.psf_stamp_integration import PsfStampIntegration
    xp = np if device < 0 else specula.cp
    params = SimulParams(time_step=.002)
    source = BaseValue(value=xp.zeros((8, 8)), target_device_idx=device)
    full = WindowedIntegration(params, [8, 8], .006, start_time=.004,
                               update_time_on_dt=True, target_device_idx=device)
    small = PsfStampIntegration(params, 8, stamp, .006, start_time=.004,
                               target_device_idx=device)
    for obj in (full, small):
        obj.inputs['input'].set(source)
        obj.setup()
    retained = None
    for step in range(8):
        source.value[:] = xp.arange(64).reshape(8, 8) + step
        source.value /= source.value.sum()
        tick = step * small.loop_dt
        source.generation_time = tick
        for obj in (full, small):
            obj.check_ready(tick)
            obj.trigger()
            obj.post_trigger()
        if step in (4, 7):
            np.testing.assert_allclose(cpuArray(small.output.value),
                                       cpuArray(full.output.value[small.stamp_slice]), rtol=2e-6)
            assert small.output.generation_time == tick
            retained = cpuArray(small.output.value, force_copy=True)
            assert retained.sum() <= 1 + 1e-6
        elif step >= 5:
            np.testing.assert_array_equal(cpuArray(small.output.value), retained)
            assert small.output.generation_time == 4 * small.loop_dt
    assert small.output.value.shape == small.integrated_value.shape == (stamp, stamp)
