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
