"""Numerical and lifecycle checks for the optional dataset fast paths."""

import logging
from contextlib import nullcontext
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


@pytest.mark.parametrize('device', DEVICES)
def test_disabled_psf_logging_never_formats_device_scalar(monkeypatch, device):
    import specula.processing_objects.dataset_psf as module

    class DeviceScalar(np.ndarray):
        def __format__(self, spec):
            raise AssertionError('Device scalar copied to host')

        def __str__(self):
            raise AssertionError('Device scalar copied to host')

    obj = DatasetPSF(SimulParams(pixel_pupil=8, pixel_pitch=.1), 1250., nd=2.,
                     target_device_idx=device)
    ef = ElectricField(8, 8, .1, target_device_idx=device)
    obj.inputs['in_ef'].set(ef)
    obj.setup()
    obj.ref.i[:] = 1
    # Indexing a subclass returns a NumPy scalar, so return a device-like 0D
    # array explicitly, as CuPy does for scalar indexing.
    class DeviceImage:
        def __getitem__(self, key):
            return np.asarray(.5).view(DeviceScalar)

    if device < 0:
        monkeypatch.setattr(module, 'calc_psf', lambda *a, **k: (DeviceImage(), 1.))
        no_host_copy = nullcontext()
    else:
        # Exercise the real GPU calculation. Scalar formatting or an explicit
        # device-to-host copy must fail while synchronization is forbidden.
        import cupyx
        no_host_copy = cupyx.allow_synchronize(False)
    obj.logger = Mock()
    obj.logger.isEnabledFor.return_value = False
    with no_host_copy:
        obj.trigger_code()
    assert isinstance(obj.sr.value, obj.xp.ndarray)
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


@pytest.mark.parametrize('device', DEVICES)
def test_psf_only_mode_matches_full_psf_without_statistics_buffers(device):
    xp = np if device < 0 else specula.cp
    params = SimulParams(pixel_pupil=8, pixel_pitch=.1)
    ef = ElectricField(8, 8, .1, target_device_idx=device)
    standard = DatasetPSF(params, 1250., nd=2., target_device_idx=device)
    fast = DatasetPSF(params, 1250., nd=2., target_device_idx=device, compute_statistics=False)
    for obj in (standard, fast):
        obj.inputs['in_ef'].set(ef)
        obj.setup()
    for step in range(3):
        ef.phaseInNm[:] = xp.arange(64).reshape(8, 8) * step
        ef.generation_time = step * fast.seconds_to_t(.002)
        for obj in (standard, fast):
            no_host_copy = nullcontext()
            if obj is fast and device >= 0:
                import cupyx
                no_host_copy = cupyx.allow_synchronize(False)
            with no_host_copy:
                obj.check_ready(ef.generation_time)
                obj.trigger()
                obj.post_trigger()
        assert isinstance(fast.psf.value, xp.ndarray)
        np.testing.assert_allclose(cpuArray(fast.psf.value), cpuArray(standard.psf.value))
        assert fast.psf.generation_time == ef.generation_time
    fast.finalize()
    assert fast.ref is fast._sum_psf_squared is None
    for name in ('int_psf', 'std_psf', 'sr', 'int_sr'):
        assert getattr(fast, name).value is None
    with pytest.raises(ValueError, match='Profile metrics require'):
        DatasetPSF(params, 1250., nd=2., compute_statistics=False, compute_profile_metrics=True)


@pytest.mark.parametrize('device', DEVICES)
def test_long_exposure_from_short_means_matches_direct_average(device):
    from specula.base_value import BaseValue
    from specula.processing_objects.completed_exposure_integration import CompletedExposureIntegration
    xp = np if device < 0 else specula.cp
    params = SimulParams(time_step=.002)
    source = BaseValue(value=xp.zeros((3, 3)), target_device_idx=device)
    obj = CompletedExposureIntegration(params, [3, 3], 100, 5000, start_time=.004,
                                       target_device_idx=device)
    obj.inputs['input'].set(source)
    obj.setup()
    rng = np.random.default_rng(13)
    for window in range(2):
        frames = rng.random((5000, 3, 3))
        for exposure in range(50):
            source.value[:] = xp.asarray(frames[exposure * 100:(exposure + 1) * 100].mean(axis=0))
            tick = (2 + window * 5000 + (exposure + 1) * 100 - 1) * obj.loop_dt
            source.generation_time = tick
            obj.check_ready(tick)
            obj.trigger()
            obj.post_trigger()
            if exposure < 49:
                assert obj.output.generation_time < tick
        np.testing.assert_allclose(cpuArray(obj.output.value), frames.mean(axis=0), rtol=2e-6)
        assert obj.output.generation_time == tick
    source.generation_time = tick + 200 * obj.loop_dt
    obj.check_ready(source.generation_time)
    with pytest.raises(ValueError, match='Missing or out-of-order'):
        obj.trigger()
    with pytest.raises(ValueError, match='positive multiple'):
        CompletedExposureIntegration(params, [3, 3], 100, 250)


def test_nondivisible_long_exposures_keep_direct_accumulation(tmp_path):
    from test.test_start_scao_dataset import physical_config, workflow
    params, settings = physical_config(tmp_path)
    settings.update(science_time=.6, long_exposure=.3)
    params['main']['total_time'] = .604
    workflow.add_direct_dataset(params, settings, tmp_path)
    assert params['exposure_J_long_0']['class'] == 'PsfStampIntegration'
    assert params['exposure_J_long_0']['inputs']['input'] == 'psf_j.out_psf'


@pytest.mark.parametrize('device', DEVICES)
def test_device_window_capture_matches_legacy_and_consolidates_transfers(tmp_path, device):
    from unittest.mock import patch
    import specula.processing_objects.scao_webdataset_store as module
    from test.test_start_scao_dataset import direct_inputs, step_capture, read_shards
    legacy = direct_inputs(tmp_path / 'legacy', device)
    buffered = direct_inputs(tmp_path / 'buffered', device)
    buffered[0].device_buffering = True
    for items in (legacy, buffered):
        items[0].setup()
    transfers = []
    try:
        for step in range(202):
            for items in (legacy, buffered):
                *objects, psfs = items
                store = objects[0]
                if step in (101, 201):
                    for key, values in psfs.items():
                        if 'short' in key or step == 201:
                            for value in values:
                                value.generation_time = step * store.step
                with patch.object(module, 'cpuArray', wraps=specula.cpuArray) as to_cpu:
                    step_capture(*objects, step)
                    if items is buffered:
                        transfers.extend(call.args[0].shape for call in to_cpu.call_args_list)
            if step == 100:
                assert transfers == [(4, 4)]  # Static pupil only, no per-step telemetry copies.
        for items in (legacy, buffered):
            items[0].finalize()
    finally:
        for items in (legacy, buffered):
            items[0].abort()
    assert transfers.count((5, 4, 4)) == 6  # Two bands, two short exposures, one long.
    assert transfers.count((100, 4, 4)) == 2  # Phase, once per short window.
    assert transfers.count((100, 2, 3)) == 2  # Detector, once per short window.
    for suffix in ('', 'long_exposures'):
        _, expected = read_shards(legacy[0].store_dir / suffix)
        _, actual = read_shards(buffered[0].store_dir / suffix)
        for a, b in zip(actual, expected):
            for key in a:
                if key.endswith('.npy'):
                    np.testing.assert_allclose(a[key], b[key])


@pytest.mark.parametrize('problem', ['pupil', 'nonfinite', 'stale_frame'])
def test_device_capture_validates_before_publishing(tmp_path, problem):
    from test.test_start_scao_dataset import direct_inputs, step_capture
    *objects, psfs = direct_inputs(tmp_path / 'dataset')
    store, commands, applied, pixels, ef = objects
    store.device_buffering = True
    store.setup()
    try:
        step_capture(*objects, 2)
        if problem == 'stale_frame':
            commands.generation_time = ef.generation_time = 3 * store.step
            applied.generation_time = 2 * store.step
            store.check_ready(3 * store.step)
            with pytest.raises(ValueError, match='current simulation step'):
                store.trigger()
        else:
            for step in range(3, 101):
                if step == 20:
                    if problem == 'pupil':
                        ef.A[0, 0] = 0
                    else:
                        commands.value[:] = np.nan
                if problem == 'nonfinite' and step == 20:
                    # step_capture replaces commands, so inject NaN in the pixels
                    # after setting the normal source values.
                    step_capture(*objects, step)
                    store._device_buffers['shwfs_frames'][store.frame - 1, 0, 0] = np.nan
                else:
                    step_capture(*objects, step)
                if problem == 'pupil' and step == 20:
                    ef.A[0, 0] = 1  # Detect even a transient change.
            for values in psfs.values():
                for value in values:
                    value.generation_time = 101 * store.step
            with pytest.raises(ValueError, match='static on-axis pupil|Invalid or changing'):
                step_capture(*objects, 101)
    finally:
        store.abort()
    assert not list(store.store_dir.rglob('*.tar'))
    assert not list(store.store_dir.rglob('dataset.json'))


@pytest.mark.parametrize('device_buffering', [False, True])
def test_background_writer_owns_snapshots_and_drains_both_streams(tmp_path, device_buffering):
    import threading
    from unittest.mock import patch
    from test.test_start_scao_dataset import direct_inputs, step_capture, read_shards
    *objects, psfs = direct_inputs(tmp_path / 'dataset')
    store = objects[0]
    store.background_writes = True
    store.device_buffering = device_buffering
    store.setup()
    entered, release = threading.Event(), threading.Event()
    write = store.streams['short'].write

    def blocked_write(sample):
        entered.set()
        assert release.wait(5), 'Test worker was not released'
        write(sample)

    try:
        with patch.object(store.streams['short'], 'write', side_effect=blocked_write):
            for step in range(202):
                if step in (101, 201):
                    for key, values in psfs.items():
                        if 'short' in key or step == 201:
                            for value in values:
                                value.generation_time = step * store.step
                if step == 102:
                    assert entered.wait(5)
                    # Mutate the live sources while the first sample is pending.
                    for values in psfs.values():
                        for value in values:
                            value.value[:] *= 2
                    # Release before the second short+long pair needs the queue.
                    release.set()
                step_capture(*objects, step)
            store.finalize()
    finally:
        release.set()
        store.abort()
    assert not store._background.thread.is_alive()
    _, samples = read_shards(store.store_dir)
    _, long_samples = read_shards(store.store_dir / 'long_exposures')
    assert len(samples) == 2 and len(long_samples) == 1
    np.testing.assert_array_equal(samples[0]['shwfs_frames.npy'][:, 0, 0], np.arange(2, 102))
    np.testing.assert_array_equal(samples[1]['shwfs_frames.npy'][:, 0, 0], np.arange(102, 202))
    assert samples[0]['psf_j.npy'].sum() == pytest.approx(1.25)
    assert samples[1]['psf_j.npy'].sum() == pytest.approx(2.5)


def test_background_queue_is_bounded_and_propagates_worker_failure():
    import threading
    from specula.processing_objects.scao_webdataset_store import _BackgroundWriter
    entered, release, third_returned = threading.Event(), threading.Event(), threading.Event()

    class SlowStream:
        def write(self, sample):
            entered.set()
            assert release.wait(5)
            raise OSError('disk full')

    writer = _BackgroundWriter()
    errors = []
    try:
        writer.submit(SlowStream(), {'x.npy': np.zeros(4)})
        assert entered.wait(5)
        writer.submit(SlowStream(), {'x.npy': np.zeros(4)})

        def submit_third():
            try:
                writer.submit(SlowStream(), {'x.npy': np.zeros(4)})
            except OSError as error:
                errors.append(str(error))
            finally:
                third_returned.set()

        producer = threading.Thread(target=submit_third)
        producer.start()
        assert not third_returned.wait(.05)
        assert writer.queue.qsize() == writer.queue.maxsize == 1
        release.set()
        assert third_returned.wait(5)
        producer.join()
        with pytest.raises(OSError, match='disk full'):
            writer.close()
        assert errors == ['disk full']
    finally:
        release.set()
        writer.close(abort=True)
    assert not writer.thread.is_alive()


@pytest.mark.parametrize('failure', ['write', 'encoding', 'interruption'])
def test_background_failure_aborts_without_manifests_or_worker_leaks(tmp_path, failure):
    from unittest.mock import patch
    import specula.processing_objects.scao_webdataset_store as module
    from test.test_start_scao_dataset import direct_inputs, step_capture
    *objects, psfs = direct_inputs(tmp_path / 'dataset')
    store = objects[0]
    store.background_writes = True
    store.setup()
    for key, values in psfs.items():
        for value in values:
            value.generation_time = 101 * store.step
    try:
        for step in range(101):
            step_capture(*objects, step)
        if failure == 'write':
            target, name = store.streams['short'].writer, 'write'
        else:
            target, name = module, 'npy_bytes'
        if failure == 'interruption':
            step_capture(*objects, 101)
            store.abort()
        else:
            with patch.object(target, name, side_effect=OSError('worker failed')):
                with pytest.raises(OSError, match='worker failed'):
                    step_capture(*objects, 101)
                    store._background.close()
    finally:
        store.abort()
    assert not store._background.thread.is_alive()
    assert all(s.closed for s in store.streams.values())
    assert not list(store.store_dir.rglob('dataset.json'))
    assert not list(store.store_dir.rglob('*.tar'))
