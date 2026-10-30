"""Import a MODE_Y_X OPD basis into SPECULA without synthesizing new modes.

Spatial coordinates are centered physical pixel coordinates. The pupil is
sampled with nearest-neighbor interpolation; mode values use mask-normalized
bilinear interpolation to avoid treating obscured pixels as zero OPD samples.
"""
from pathlib import Path
import numpy as np
from astropy.io import fits
from scipy.ndimage import map_coordinates


def resample_modes(cube, pupil, source_pitch_yx, target_size, target_pitch):
    """Return [mode, illuminated target pixel] OPD maps and a binary pupil.

    No fit, reordering or amplitude renormalization is performed. Masked
    samples are excluded from bilinear interpolation, including at spiders.
    """
    pupil = np.asarray(pupil)
    if cube.ndim != 3 or cube.shape[1:] != pupil.shape:
        raise ValueError("Expected mode cube [mode,y,x] and matching 2-D PUPIL")
    if not np.all((pupil == 0) | (pupil == 1)) or not pupil.any():
        raise ValueError("PUPIL must be non-empty and binary")
    pitches = np.asarray(source_pitch_yx, dtype=float)
    if pitches.shape != (2,) or not np.isfinite(pitches).all() or (pitches <= 0).any():
        raise ValueError("Source pixel pitches must be finite and positive")
    if target_size < 2 or not np.isfinite(target_pitch) or target_pitch <= 0:
        raise ValueError("Invalid target grid")
    axis = (np.arange(target_size) - (target_size - 1) / 2) * target_pitch
    y, x = np.meshgrid(axis / pitches[0] + (pupil.shape[0] - 1) / 2,
                       axis / pitches[1] + (pupil.shape[1] - 1) / 2, indexing="ij")
    if y.min() < 0 or x.min() < 0 or y.max() > pupil.shape[0]-1 or x.max() > pupil.shape[1]-1:
        raise ValueError("Target pixel centers exceed the source map; extrapolation is disabled")
    coords = np.array([y, x])
    target_mask = map_coordinates(pupil.astype(float), coords, order=0, mode="constant") > 0.5
    if not target_mask.any():
        raise ValueError("Resampled pupil is empty")
    coverage = map_coordinates(pupil.astype(float), coords, order=1, mode="constant")
    if np.any(coverage[target_mask] <= 0):
        raise ValueError("Target pupil includes a pixel with no valid source samples")
    result = np.empty((cube.shape[0], int(target_mask.sum())), dtype=np.float64)
    for i, mode in enumerate(cube):
        if not np.isfinite(mode[pupil > 0]).all():
            raise ValueError(f"Mode {i} contains non-finite illuminated values")
        valid_values = np.where(pupil > 0, mode, 0).astype(float)
        sampled = map_coordinates(valid_values, coords, order=1, mode="constant")
        result[i] = sampled[target_mask] / coverage[target_mask]
    return result, target_mask.astype(np.uint8)


def import_basis(path, cfg, mode_start=0):
    """Validate file metadata, import the requested modes and report provenance."""
    path = Path(path)
    nmodes = cfg["dm"]["nmodes"]
    npx, pitch = cfg["main"]["pixel_pupil"], cfg["main"]["pixel_pitch"]
    with fits.open(path, memmap=True) as hdul:
        hdr = hdul[0].header
        if hdr.get("LAYOUT") != "MODE_Y_X" or hdr.get("BUNIT", "").strip().lower() != "nm":
            raise ValueError("Expected LAYOUT=MODE_Y_X and BUNIT=nm (OPD)")
        if hdr.get("NORM") != "PUPIL_RMS_1NM":
            raise ValueError("Expected supplied 1 nm pupil-RMS OPD normalization")
        cube, pupil = hdul[0].data, hdul["PUPIL"].data
        if cube is None or cube.ndim != 3 or mode_start < 0 or mode_start + nmodes > cube.shape[0]:
            raise ValueError("Source cube does not contain the selected mode range")
        sx, sy = float(hdr["PIXX_M"]), float(hdr["PIXY_M"])
        diameter = float(hdr["DIAM_M"])
        if not np.isclose(npx*pitch, diameter, rtol=0, atol=1e-9):
            raise ValueError("Simulation aperture must match source DIAM_M")
        if not np.allclose(np.asarray(pupil.shape)*[sy,sx], diameter, rtol=0, atol=1e-9):
            raise ValueError("Source pupil sampling is inconsistent with DIAM_M")
        selected = cube[mode_start:mode_start+nmodes]
        source_values = np.asarray(selected[:, pupil > 0], dtype=np.float64)
        source_rms = np.sqrt(np.mean(source_values**2, axis=1))
        if not np.allclose(source_rms, 1.0, atol=1e-4, rtol=0):
            raise ValueError("Selected maps do not have the declared 1 nm pupil RMS")
        maps, mask = resample_modes(selected, pupil, (sy, sx), npx, pitch)
        if np.any(np.std(maps, axis=1) < 1e-8):
            raise ValueError("Resampling produced a zero/piston-only mode")
        labels = np.asarray(hdul["NOLL"].data[mode_start:mode_start+nmodes], dtype=int)
        target_rms = np.sqrt(np.mean(maps**2, axis=1))
        return maps, mask, {
            "basis_kind": "empirical modal OPD maps; no actuator M2C",
            "source_file": str(path), "source_shape": list(cube.shape),
            "source_diameter_m": diameter, "source_pitch_yx_m": [sy,sx],
            "source_norm": hdr["NORM"], "source_r0_m": float(hdr["R0_M"]),
            "source_L0_m": float(hdr["L0_M"]), "source_piston": hdr.get("PISTON"),
            "source_zernike_modes": int(hdr["NZERN"]), "source_kl_modes": int(hdr["NKL"]),
            "selected_mode_start": mode_start, "controlled_modes": nmodes,
            "selected_zernike_noll": labels[labels != 0].tolist(),
            "selected_kl_modes": int(np.count_nonzero(labels == 0)),
            "target_pupil_pixels": int(mask.sum()), "target_size": npx,
            "target_pitch_m": pitch, "target_diameter_m": npx*pitch,
            "target_rms_nm_range": [float(target_rms.min()),float(target_rms.max())],
            "target_abs_mean_nm_max": float(np.abs(maps.mean(axis=1)).max()),
            "resampling": "centered physical coordinates; nearest mask; mask-normalized bilinear OPD",
            "renormalized": False, "extra_dm_registration_applied": False,
        }


def main():
    """Tutorial preprocessing step: convert supplied maps to native FITS objects."""
    import argparse
    import sys
    import yaml
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("calib/atm_dm_kl.fits"))
    parser.add_argument("--root", type=Path, help="Optional calibration-root override")
    parser.add_argument("--precision", type=int, choices=[0, 1], default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load((here / "params_truthwfs_scao.yml").read_text())
    def resolve(path):
        return path.expanduser().resolve() if path.is_absolute() else (repo / path).resolve()
    source = resolve(args.source)
    root = resolve(args.root or Path(cfg["main"]["root_dir"]))
    sys.path.insert(0, str(repo))

    import specula
    specula.init(device_idx=-1, precision=args.precision)
    from specula.calib_manager import CalibManager
    from specula.data_objects.ifunc import IFunc
    cm = CalibManager(str(root))
    products = [Path(cm.filename("ifunc", cfg["dm"]["ifunc_object"])),
                Path(cm.filename("ifunc_inv", cfg["residual_modes"]["ifunc_inv_object"])),
                Path(cm.filename("data", cfg["pupilstop"]["input_mask_data"]))]
    if not args.overwrite:
        for path in products:
            if path.exists():
                raise FileExistsError(f"{path} already exists; use --overwrite to reconvert")
    # These maps already represent modal OPD; no synthetic IF/M2C generation.
    maps, mask, info = import_basis(source, cfg, mode_start=0)
    modal = IFunc(ifunc=maps.astype([np.float64, np.float32][args.precision]), mask=mask)
    inverse = modal.inverse()
    for path in products:
        path.parent.mkdir(parents=True, exist_ok=True)
    modal.save(str(products[0]), overwrite=args.overwrite)
    inverse.save(str(products[1]), overwrite=args.overwrite)
    fits.writeto(products[2], mask, overwrite=args.overwrite)
    for path in products:
        print(f"Saved {path}")
    print(f"Imported {info['controlled_modes']} supplied modes: "
          f"{len(info['selected_zernike_noll'])} Zernike + {info['selected_kl_modes']} KL")
    print(f"Pupil: {info['target_size']} pixels, {info['target_diameter_m']} m; "
          f"resampled RMS range {info['target_rms_nm_range']}")


if __name__ == "__main__":
    main()
