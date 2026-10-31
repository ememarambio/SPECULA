# START-SCAO: J/K PSF exposures across a 21 arcsecond square

`run_psf_grid.py` runs the AO loop once and saves every DM command. It then
replays that same atmospheric and DM history in small batches of field points,
computing both J and K PSFs for each point. The WFS/controller is only run in
the science stage. Run the commands below from the repository root with a
Python environment containing SPECULA's dependencies.

Upstream now bundles the `iteras` simulation engine and adds `h5py` to its
requirements. Update the active environment after pulling with
`python -m pip install -e .` from the repository root. This workflow explicitly
saves telemetry as FITS, including when an environment variable selects another
default storage format.

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

## Science run and field replay

```sh
python config/START-SCAO/run_psf_grid.py run output/START-SCAO/run01
python config/START-SCAO/run_psf_grid.py replay output/START-SCAO/run01
```

CPU is the default. For CUDA use `--target 0` **before** `run` or `replay`;
`--precision 0` selects float64 instead of the default float32. Replay restores
the science run's recorded precision. Add `--displays` to `run` for the optional
on-axis diagnostic plots.

`psf_grid.yml` controls the science duration, settling time, exposure windows,
field sampling, batch size, PSF sampling, stamp size and wavelengths. A copy is saved with
each science run and used by replay. To use another settings file, pass
`run ... --settings PATH`. `run ... --science-time 30` records 30 seconds after
settling, producing three independent 10 s exposures. The usable duration must
contain complete short and long windows; invalid durations are rejected.

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
| Replay batch | 16 field points, both bands |
| J / K wavelength | 1,250 / 2,200 nm |
| Calculated PSF size | 512 × 512 pixels (`pixel_pupil: 256`, `nd: 2`) |
| Saved PSF stamp | Central 128 × 128 pixels (`stamp_size: 128`) |
| J / K pixel scale | Approximately 16.27 / 28.64 mas per pixel |
| J stamp FoV | Approximately 2.08″ × 2.08″ |
| K stamp FoV | Approximately 3.67″ × 3.67″ |

These are normalized monochromatic PSFs at representative band wavelengths,
not filter-integrated or detector-count images. The shared J/K OPD calculation
requires geometric propagation without chromatic shifts; the replay builder
rejects Fresnel/chromatic propagation settings. A 0.2 s finite exposure need
not be statistically converged to an infinite-exposure PSF. Check the settling
duration against the loop behavior for the actual observing conditions.

`atmo.fov` is set to the full square width before the science run so all replay
directions fit within the atmospheric layers. The atmosphere seed, parameters,
precision, commands, and `dm.out_layer:-1` delay are preserved. Each batch starts
at t=0, including settling, to reproduce the same evolving atmosphere and DM
history. Commands with missing timestamps or downsampling are rejected.

## Exposure semantics and FITS layout

Each instantaneous PSF feeds two `WindowedIntegration` objects. They average
intensity over independent, non-overlapping windows and clear their accumulators
after each completed exposure. Only completed windows update their output
timestamp. `PsfExposureStore` writes each resulting cube immediately, keeping
memory bounded as simulation duration grows. It never writes a partial window
or replaces an existing exposure. It does not use `PSF.out_int_psf`, which is
averaged only when the whole simulation finishes.

`stamp_size: 128` crops each completed exposure about the FFT center, retaining
the native pixel scale at `nd: 2.0` (two pixels per lambda/D). It does not
recenter on the brightest pixel or renormalize the crop: flux outside the
stamp is excluded, so the saved pixel sum is the retained fraction of the
full PSF's unit flux. Set `stamp_size: null` to save the full calculated image.
The same crop applies to on-axis and field PSFs in both exposure windows and
both bands. Display images continue to show the full instantaneous PSF.

The writer uses the simulation device so readiness checks read the window
timestamps without copying their image buffers. Completed stamps are cropped
on that device and then transferred to the CPU for FITS output. There are no
full-size CPU mirror buffers for the window outputs. If configuring processing
objects on different devices manually, keep each writer on its producers' device.

For example, short exposure 0 covers [0.2, 0.4), exposure 1 covers [0.4, 0.6),
and long exposure 0 covers [0.2, 10.2). The internal producer timestamp labels
the final sample (0.398 s for the first short exposure); FITS records the actual
start and exclusive end bounds. The mean of the first 50 short exposures equals
the first long exposure up to floating-point summation error.

Output structure:

```text
run01/
  comm.fits                    # Every DM input command, including settling
  params.yml                   # Actual science graph
  replay_params.yml            # Recorded precision and seeds
  science.yml                  # Generated science input
  psf_grid.yml                 # Frozen exposure/grid settings
  SCIENCE_COMPLETE             # Written only after the science stage returns
  on_axis/J/short/exposure_000000.fits
  on_axis/J/long/exposure_000000.fits
  on_axis/K/...                # Same layout
  field/grid.csv               # FIELD_ID,x_arcsec,y_arcsec for the entire grid
  field/batch_0000_0015/
    replay.yml                 # Exact batch graph
    COMPLETE.json              # Written only after the entire batch succeeds
    J/short/exposure_000000.fits
    J/long/exposure_000000.fits
    K/...
```

Each exposure's primary HDU is an array `[field, 128, 128]` by default. Its header
contains `BAND`, `WAVELEN` (nm), `PIXSCALE` (mas/pixel), `NFRAMES`, `EXPTIME`,
`TIMEDEL`, `TSTART`, and `TEND` (seconds). `FULLNX`/`FULLNY` record the original
image dimensions, `CROPX0`/`CROPY0` give the zero-based crop origin,
`FOVX`/`FOVY` give the stamp's angular width/height in arcseconds, and
`NORM=FULLPSF` records that normalization applies before cropping.
The `FIELD` table maps the first array
axis to `FIELD_ID`, `X_ARCSEC`, and `Y_ARCSEC`. Grid IDs are row-major:
`FIELD_ID = iy * grid_size + ix`; x varies fastest. The even-sized field grid
does not include (0, 0); separate on-axis diagnostics use `FIELD_ID = -1`.

For a small initial replay or a later subset:

```sh
python config/START-SCAO/run_psf_grid.py replay output/START-SCAO/run01 --point-count 16
python config/START-SCAO/run_psf_grid.py replay output/START-SCAO/run01 --first-point 16
```

Use `--batch-size 4` to lower peak memory. Existing batch directories cause an
error rather than being overwritten. After a failed batch, retain/remove its
incomplete directory as appropriate and restart at its first point. Completed
batches have `COMPLETE.json`; no automatic skip or overwrite is performed.

The default full grid produces about **6.375 GiB** of uncompressed float32 image
data per 10 s of science, plus on-axis diagnostics and small metadata files.
Float64 doubles this. Cropping reduces saved image data by a factor of 16;
the full FFTs and exposure accumulation still use 512 × 512 arrays. Batching
limits memory, not disk usage or total FFT work.
Benchmark a small batch before processing all 1,024 points.

## WebDataset training samples

Install the optional [WebDataset](https://github.com/webdataset/webdataset)
dependency, capture training telemetry in a **new** science run, finish all
field batches, then export:

```sh
python -m pip install -e '.[dataset]'
python config/START-SCAO/run_psf_grid.py --target 0 --precision 1 run output/START-SCAO/train01 --dataset
python config/START-SCAO/run_psf_grid.py --target 0 replay output/START-SCAO/train01
python config/START-SCAO/generate_webdataset.py output/START-SCAO/train01 --out shards/train01
```

Replay can use `--batch-size 1024` for the entire field at once, or the default
16 to limit GPU memory. The exporter runs on the CPU and does not rerun the
simulator. Old runs containing only `comm.fits` lack the recorded SHWFS noise
realization and phase sequence; capture a new run with `--dataset` for these
samples. The usual FITS telemetry and both exposure durations are still saved.
Dataset targets use **only the independent 100-frame exposures**; the 10 s
PSFs remain available separately as FITS.

Each sample covers exactly 100 consecutive steps after settling: 0.2 s at
500 Hz. With the defaults, the first sample covers steps 100–199, timestamps
0.200–0.398 s, and exposure interval **[0.200, 0.400) s**. The next starts at
step 200. Ten seconds of science produces 50 samples, with no overlapping
windows. `--dataset` requires `short_frames: 100` and one detector exposure
per simulation step.

Every tar sample has a unique `<run_id>_<window_index>` key and these members:

| Member | Default shape | Meaning |
| --- | --- | --- |
| `dm_commands.npy` | `[100, 702]` | DM coefficients actually applied during each SHWFS exposure, before the DM's `sign: -1` |
| `controller_commands.npy` | `[100, 702]` | Controller output emitted at those same simulation steps |
| `shwfs_frames.npy` | `[100, 224, 224]` | Actual detector frames, including the configured noise, in native detector units |
| `residual_phase.npy` | `[100, 256, 256]` | On-axis residual phase in unwrapped radians at 1,250 nm (the configured J wavelength) |
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

Residual phase keeps piston and is not wrapped modulo 2π. Convert it to K-band
radians with `phase_k = residual_phase * wavelength_j_nm / wavelength_k_nm`, or
to wavelength-independent OPD in nm with
`opd_nm = residual_phase * wavelength_j_nm / (2 * np.pi)`. The saved pupil must
be used when computing a PSF from phase. The phase sequence is on-axis only;
the PSF targets include the full off-axis field. Stamps retain their native
J/K pixel scales and full-image normalization; cropping does not renormalize
their retained flux. The 32×32 field alone is
`psf_j[1:].reshape(32, 32, 128, 128)`; `psf_j[0]` is the on-axis stamp.

Capture streams `dataset_telemetry/window_XXXXXX.npz` using one 100-frame host
buffer (about 45 MiB in float32), plus the existing small command history for
replay. Export assembles one complete sample at a time. At the defaults, a
sample is approximately **173 MiB**, or **8.4 GiB per 10 s** of uncompressed
shards, in addition to the source telemetry/FITS files. Float64 roughly doubles
this. Shards contain at most four samples by default; `--shard-size` changes
this limit and `--max-shard-gib` sets a size threshold (rotation happens between
samples, so the threshold can be exceeded by one sample).

The exporter requires complete telemetry and complete, non-overlapping field
batches. It checks integer timestamps, controller values against `comm.fits`,
exposure bounds, field coordinates, wavelengths, crop geometry, and precision
across batches. It refuses existing destinations and assembles into a temporary
directory, publishing the requested directory only after all samples succeed.
`dataset.json` lists shards and sample counts; `runs/<run_id>/` holds the source
configuration and telemetry metadata. No calibration FITS files are packaged.
Multiple complete runs can be passed before `--out`; duplicate run IDs are
rejected. For train/validation splits, keep entire independent runs together
to avoid sharing adjacent, temporally correlated windows across splits.
The base configuration uses fixed random seeds: repeating it unchanged gives
the same simulation, even with a new run ID. Vary the atmosphere and detector
seeds in the science configuration when generating independent realizations.

Example loader (install PyTorch separately if needed for training):

```python
import json
from pathlib import Path
import webdataset as wds

root = Path('shards/train01')
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
