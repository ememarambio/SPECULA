# START-SCAO: synchronized J/K WebDataset generation

`run_psf_grid.py` runs the AO loop and the entire science field together,
streaming 100-step training samples and independent long-exposure PSFs directly
to WebDataset. Run the commands below from the repository root.

SPECULA includes the `iteras` engine. Install the dataset extra below to update
the environment and include WebDataset support.

## Calibration

The empirical input `calib/atm_dm_kl.fits` is a local prerequisite and is
deliberately excluded from Git. Neither that file nor generated calibration
products are included in this configuration. Import the supplied basis, then
run the three native calibration benches in order:

```sh
python config/START-SCAO/compute_influence_functions.py --source calib/atm_dm_kl.fits
python -m specula.scripts.specula_main --cpu config/START-SCAO/params_truthwfs_scao.yml config/START-SCAO/calib_subaps.yml
python -m specula.scripts.specula_main --cpu config/START-SCAO/params_truthwfs_scao.yml config/START-SCAO/calib_reference.yml
python -m specula.scripts.specula_main --cpu config/START-SCAO/params_truthwfs_scao.yml config/START-SCAO/calib_im_rec.yml
```

The calibration root is `calib/ULTIMATE_SCAO_EMPIRICAL`. The importer refuses
to replace existing products unless `--overwrite` is passed. The calibration
YAML benches have `overwrite: true`. After updating from the older SHWFS model,
rerun all three benches above: upstream's exact detector-pixel integration
changes the sensor images, slope references, interaction matrices and
reconstructors. Previously generated SH calibration products must be replaced.
The empirical basis does not need to be reimported if it is already present.
Displays are in `display.yml`, separate from these benches.

## WebDataset training samples

Install the optional [WebDataset](https://github.com/webdataset/webdataset)
dependency, then run the full dataset simulation:

```sh
python -m pip install -e '.[dataset]'
python config/START-SCAO/run_psf_grid.py --target 0 --precision 1 run output/START-SCAO/train01
```

CPU is the default. For CUDA use `--target 0` before `run`; `--precision 1`
selects float32 and `--precision 0` selects float64. Add `--displays` for the
optional on-axis diagnostic plots.

`psf_grid.yml` controls science and settling durations, exposure windows, field
sampling, stamp size, and J/K wavelengths. Pass `--settings PATH` to use another
settings file. `--science-time 30` generates 30 seconds after settling: 150
training samples and three independent 10-second exposures. The duration must
contain complete short and long windows. Each run saves a frozen settings copy.

## Reference and intermediate profile datasets

Two ready-to-run configurations use Table 1 of Oya et al. (2014),
*ULTIMATE-SUBARU: simulation update*, [doi:10.1117/12.2055701](https://doi.org/10.1117/12.2055701).
There is **no pilot stage and no assigned train/validation/test split**.

| Settings file | Observations | Definition |
| --- | ---: | --- |
| `reference_profile_dataset.yml` | 450 (indices 0–449) | 3 complete seeing-conditioned profiles × 5 zenith angles × 3 wind speed scales × 10 realizations |
| `intermediate_profile_dataset.yml` | 150 (indices 0–149) | 75 interpolations between good/moderate and 75 between moderate/bad |

Each observation has **1 s settling plus 10 s science at 500 Hz**. The science
interval is [1, 11) s: 5,000 steps, 50 nonoverlapping 100-frame samples, and one
10-second exposure. Both configurations retain the full 32 × 32 square grid
plus on-axis, J/K, `nd: 2`, and 128 × 128 stamps. Across both datasets this gives
30,000 short-window samples and 600 long-exposure samples.

Resolve and inspect every condition without initializing SPECULA or a GPU:

```sh
mkdir -p output
python config/START-SCAO/run_psf_grid.py manifest --settings config/START-SCAO/reference_profile_dataset.yml > output/reference_observations.jsonl
python config/START-SCAO/run_psf_grid.py manifest --settings config/START-SCAO/intermediate_profile_dataset.yml > output/intermediate_observations.jsonl
```

Run one observation per process/GPU job using the existing direct WebDataset path:

```sh
python config/START-SCAO/run_psf_grid.py --target 0 --precision 1 run output/reference/obs_000000 --settings config/START-SCAO/reference_profile_dataset.yml --observation 0
python config/START-SCAO/run_psf_grid.py --target 0 --precision 1 run output/intermediate/obs_000000 --settings config/START-SCAO/intermediate_profile_dataset.yml --observation 0
```

Change both the index and output directory for each observation. For a cluster
job array, substitute the array task index for `--observation` and include it in
the output directory. Use array indices 0–449 for reference and 0–149 for
intermediate. Each invocation generates exactly one observation; no GPU batching
or replay is involved. Existing output directories are rejected rather than
overwritten. The original single-observation `psf_grid.yml` still works without
`--observation`. Campaign settings require an explicit index and exactly 10 s of
science; `--science-time` cannot extend a campaign observation.

`oya2014_profiles.yml` records the seven layer heights, fractional contributions,
and absolute integrated strengths from the paper. Total strengths are
3.5781e-13, 5.2770e-13, and 8.1360e-13 m^(1/3). The atmospheric outer scale is
30 m for these datasets; the original single-observation config uses 20 m.
The zero-height contribution already includes the paper's assumed local/telescope
contribution. No extra dome layer is added. The instrument calibration, empirical
DM basis and controller are retained.

For each reference profile, zenith angles are 0, 15, 30, 45 and 60 degrees and
wind speed scales are 0.5, 1 and 2. The ten realizations in each cell use one
jittered global wind rotation per 36-degree stratum, shuffled deterministically.
The paper's wind speeds are MAOS defaults and its 50-degree layer direction
increments are artificial. These wind variations are coverage assumptions, not
measured Subaru weather frequencies.

Intermediate profiles interpolate **absolute layer strengths**, then derive total
strength, normalized fractions and r0. The interpolation coefficients are shuffled
midpoints of 75 equal strata in (0, 1) for each adjacent pair. Zenith angle,
wind speed scale and global rotation are sampled independently and uniformly over
[0, 60] degrees, [0.5, 2], and [0, 360) degrees, respectively. These are synthetic
interpolations within the paper's reference cases, not estimated probabilities.

The runner derives r0 at 500 nm from the absolute turbulence integral and converts
it to SPECULA's seeing parameter (approximately 0.676, 0.853 and 1.106 arcsec for
the reference cases). The paper's finite-outer-scale FWHM values must not be used
as `seeing.constant`. Strengths/heights remain referenced to zenith; SPECULA applies
airmass once.

Both configurations use 16,384-pixel screens (about 507 m at the configured pitch).
The maximum wind travels about 419 m during 11 s. Before simulation, the runner
checks the screen length against travel distance plus a conservative field
footprint and rejects configurations that would cycle the screens.

Every observation has a stable `observation_id`, distinct atmospheric and detector
seeds, and a resolved `observation.json` beside the frozen simulation graph. The
same record is included under `observation` in dataset metadata and each short-
and long-exposure shard's `meta.json`. Seed blocks are disjoint between the two
provided configurations. Model training code decides the splits; keep all windows
from an observation together when avoiding temporal leakage. Copying a campaign
to create additional independent observations requires a new simulation seed range.

Float32 storage is approximately 9.2 GB per observation, or 5.5 TB for the complete
600-observation set, excluding cached phase screens and calibration files.

## Original single-observation defaults

Defaults:

| Setting | Value |
| --- | --- |
| Simulation step | 0.002 s (500 Hz) |
| Settling interval | [0, 0.2) s |
| Usable science interval | [0.2, 10.2) s |
| Short exposure | 100 frames = 0.2 s; 50 images per point/band |
| Long exposure | 5,000 frames = 10 s; one image per point/band |
| Field | 21″ total width on each axis, from −10.5″ to +10.5″ |
| Grid | 32 × 32 points, including the boundaries; spacing 21/31 ≈ 0.677″ |
| J / K wavelength | 1,250 / 2,200 nm |
| Calculated PSF size | 512 × 512 pixels (`pixel_pupil: 256`, `nd: 2`) |
| Saved PSF stamp | Central 128 × 128 pixels (`stamp_size: 128`) |
| J / K pixel scale | Approximately 16.27 / 28.64 mas per pixel |
| J stamp FoV | Approximately 2.08″ × 2.08″ |
| K stamp FoV | Approximately 3.67″ × 3.67″ |

These are normalized monochromatic PSFs at representative band wavelengths,
not filter-integrated or detector-count images. The shared J/K OPD calculation
requires geometric propagation without chromatic shifts; the dataset builder
rejects Fresnel/chromatic propagation settings. A 0.2 s finite exposure need
not be statistically converged to an infinite-exposure PSF. Check the settling
duration against the loop behavior for the actual observing conditions.

`atmo.fov` covers the full square field. All directions use the same evolving
atmosphere and delayed DM state in the live AO loop.

This computes **all 1,024 grid points plus on-axis in both J and K during the
same AO simulation**. Each completed
100-step sample is queued for writing through `webdataset.ShardWriter`;
there is no replay, FITS/NPZ staging, command-history file, or export pass.
The full PSF calculations and cropped exposure buffers must fit on the selected device.
Use float32 (`--precision 1`, the default) for the planned 40 GB A100 run.
The full-size GPU run has not been profiled; validation uses small CPU fields.

The dataset uses `DatasetPSF` for lazy GPU diagnostic logging and
`PsfStampIntegration` to crop each normalized PSF before exposure accumulation.
Dataset PSFs disable unused Strehl and cumulative statistics with
`compute_statistics: false`; the two on-axis PSFs retain statistics when
`--displays` requests the Strehl plot. The default `DatasetPSF` API still
provides all standard PSF outputs.
The original `PSF` and `WindowedIntegration` classes remain available. The stamp
accumulators reset after each completed window; their output timestamp updates
only at an exposure boundary and their output retains the last completed image.
When the long duration is a multiple of 100 frames,
`CompletedExposureIntegration` averages completed short exposures to produce the
long target (50 short means for a 10-second exposure). Otherwise the long target
uses its own stamp accumulator. Grouping averages can change floating-point
rounding, but preserves the exposure interval and intensity mean.

The dataset workflow opts into `ScaoWebdatasetStore.device_buffering`: telemetry
is snapshotted on the simulation device for 100 steps and copied to the host at
the window boundary. Each PSF band is transferred as one field cube. Timestamp
and shape checks still run every step; static-pupil and finite-value checks are
resolved before publishing each window. This adds about 45 MiB of device
telemetry storage with the default float32 shapes. The writer defaults to its
original per-frame capture and per-field transfers when used independently.

The workflow also opts into `background_writes`: one host worker encodes NumPy
arrays and writes both shard streams while the simulation continues. Snapshots
own their data, and the queue holds at most one waiting sample in addition to
the active write. A full queue applies backpressure. Worker failures propagate
to the simulation; finalization drains the queue before closing shards and
publishing manifests. Abort joins the worker and retains partial-shard marking.
The standalone writer default remains synchronous.

Only completed windows are written. For the default timing, short exposure 0 covers [0.2, 0.4), short
exposure 1 covers [0.4, 0.6), and long exposure 0 covers [0.2, 10.2).

`stamp_size: 128` crops about the FFT center at the native `nd: 2.0` sampling.
The crop keeps its original flux: the saved sum is the retained fraction of
full-PSF unit flux. Set `stamp_size: null` for full calculated images. Cropping
applies to both bands, all field positions, and both exposure durations.

Training targets use the independent 100-frame exposures. The independent
10-second J/K exposures are saved directly to a **second WebDataset stream**
under `dataset/long_exposures/`. These samples contain `psf_j.npy`, `psf_k.npy`,
field IDs/coordinates, and metadata, with the same field order and stamp sizes.
They do not repeat telemetry; their metadata specifies the full exposure's
`start_tick`, `end_tick`, and `n_frames` (5,000 by default).

```text
train01/
  science.yml                 # Generated full simulation graph
  psf_grid.yml                # Frozen settings
  DATASET_COMPLETE            # Written only after the simulation succeeds
  dataset/
    params.yml                # Actual simulation parameters
    metadata.json             # Run ID, precision, timing, units, field sampling
    data-000000.tar            # Training samples, at most four per shard by default
    data-000001.tar
    dataset.json              # Training shard names and sample counts
    long_exposures/
      data-000000.tar          # Independent long J/K exposures
      dataset.json            # Long-exposure shard names and sample counts
```

Each sample covers exactly 100 consecutive steps after settling: 0.2 s at
500 Hz. With the defaults, the first sample covers steps 100–199, timestamps
0.200–0.398 s, and exposure interval **[0.200, 0.400) s**. The next starts at
step 200. Ten seconds of science produces 50 samples, with no overlapping
windows. Generation requires `short_frames: 100` and one detector exposure
per simulation step.

Every tar sample has a unique `<run_id>_<window_index>` key and these members:

| Member | Default shape | Meaning |
| --- | --- | --- |
| `dm_commands.npy` | `[100, 702]` | DM coefficients actually applied during each SHWFS exposure, before the DM's `sign: -1` |
| `controller_commands.npy` | `[100, 702]` | Controller output emitted at those same simulation steps |
| `shwfs_frames.npy` | `[100, 224, 224]` | Actual detector frames, including the configured noise, in native detector units |
| `residual_phase.npy` | `[100, 256, 256]` | On-axis residual phase in unwrapped radians at 1,250 nm (the configured J wavelength), with pupil-mean piston removed independently per frame |
| `pupil.npy` | `[256, 256]` | Static pupil amplitude; phase is zero outside it |
| `psf_j.npy`, `psf_k.npy` | `[1025, 128, 128]` each | Mean PSFs over exactly these 100 steps; field index 0 is on-axis |
| `field_ids.npy` | `[1025]` | `[-1, 0, ..., 1023]`: on-axis followed by row-major grid IDs |
| `field_xy_arcsec.npy` | `[1025, 2]` | Matching `[x, y]` field coordinates |
| `time_ticks.npy` | `[100]` | Shared exposure/phase/controller timestamps, in integer nanoseconds |
| `dm_command_time_ticks.npy` | `[100]` | Generation timestamps of the DM commands applied in those exposures |
| `meta.json` | JSON | Units, phase reference wavelength, timing, field order, J/K sampling, and run ID |

The row at index `i` in the command, detector and phase arrays describes the
**same physical simulation step**. Propagation uses `dm.out_layer:-1`, so the
command actually shaping frame `t` is the DM output from `t − time_step`.
The capture reads `dm.out_clipped_command:-1` before the DM updates, rather
than pairing the frame with the newly emitted controller command. Both command
forms are provided explicitly. The integrator's own delay is already included
in its output; no extra shift is applied. If settling is zero, the first
applied command is the initial zero command, with generation timestamp `-1`.

Residual phase has its piston removed and is not wrapped modulo 2π. For each
of the 100 frames, the arithmetic mean over nonzero pupil pixels is subtracted;
pixels outside the pupil remain zero. This removes only the global piston,
preserving relative phase between disconnected pupil regions. The live
simulation electric field is not modified, and the PSF intensities are
unaffected by this global phase offset. This applies to every saved residual,
in both per-frame and device-buffered capture. Sample metadata records
`phase_piston_removed: true` and the phase convention.
Convert it to K-band
radians with `phase_k = residual_phase * wavelength_j_nm / wavelength_k_nm`, or
to wavelength-independent OPD in nm with
`opd_nm = residual_phase * wavelength_j_nm / (2 * np.pi)`. The saved pupil must
be used when computing a PSF from phase. The phase sequence is on-axis only;
the PSF targets include the full off-axis field. Stamps retain their native
J/K pixel scales and full-image normalization; cropping does not renormalize
their retained flux. The 32×32 field alone is
`psf_j[1:].reshape(32, 32, 128, 128)`; `psf_j[0]` is the on-axis stamp.

Capture uses one 100-frame host buffer (about 45 MiB in float32). At each
completed exposure, the writer transfers only the central PSF stamps from the
simulation device and serializes one complete sample. Readiness checks do not
copy full PSF arrays. GPU PSF accumulators still use the full 512×512 images.
At the defaults, a training sample is approximately **173 MiB**: **8.45 GiB per
10 s**, plus about **128 MiB** for the corresponding long-exposure PSFs. There
is no duplicate intermediate dataset. Float64 roughly doubles these sizes.
Shards contain at most four samples by default; pass `--shard-size` and
`--max-shard-gib` to `run` to adjust rotation. The same limits apply independently
to both streams. Rotation happens between samples, so the byte threshold can
be exceeded by one sample.

The writer verifies current-step telemetry timestamps, the applied DM update,
and synchronized PSF completion at each exposure boundary. Incomplete windows
are never emitted. Shards are written as `.tar.partial` until closed; already
closed shards remain available after an interruption. `dataset.json` is written
only after all requested windows have been captured and both streams closed.
An interrupted run has no `DATASET_COMPLETE` marker. Restart in a new output
directory; automatic resume is not implemented. Existing directories are
refused. No calibration FITS files are packaged.

For train/validation splits, keep entire independent runs together to avoid
sharing adjacent, temporally correlated windows across splits.
The base configuration uses fixed random seeds: repeating it unchanged gives
the same simulation, even with a new run ID. Vary the atmosphere and detector
seeds in the science configuration when generating independent realizations.

Example loader (install PyTorch separately if needed for training):

```python
import json
from pathlib import Path
import webdataset as wds

root = Path('output/START-SCAO/train01/dataset')
manifest = json.loads((root / 'dataset.json').read_text())
urls = [str(root / shard['file']) for shard in manifest['shards']]
samples = (
    wds.WebDataset(urls, shardshuffle=False)
    .decode()
    .to_tuple('dm_commands.npy', 'shwfs_frames.npy', 'residual_phase.npy',
              'psf_j.npy', 'psf_k.npy', 'pupil.npy', 'meta.json')
)
commands, frames, phase, psf_j, psf_k, pupil, metadata = next(iter(samples))
```

NumPy members are serialized with pickle disabled. Keep shuffle buffers and
training batches small initially: each sample already contains 100 frames
and both complete PSF fields.
