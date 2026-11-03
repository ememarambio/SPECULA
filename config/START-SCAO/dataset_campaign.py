"""Resolve small campaign configs into reproducible, unsplit observations.

Only NumPy and YAML are needed to inspect a manifest; no simulation is started.
Each observation is run in its own process by run_psf_grid.py.
"""

from pathlib import Path

import numpy as np
import yaml


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _values(values, name, minimum, maximum, *, strictly_positive=False):
    array = np.asarray(values, dtype=float)
    if (array.ndim != 1 or not len(array) or not np.isfinite(array).all()
            or np.any(array < minimum) or np.any(array > maximum)
            or (strictly_positive and np.any(array <= 0))):
        raise ValueError(f'Invalid {name}')
    return array


def _range(values, name, minimum, maximum, **kwargs):
    array = _values(values, name, minimum, maximum, **kwargs)
    if len(array) != 2 or array[0] >= array[1]:
        raise ValueError(f'{name} must contain increasing lower and upper bounds')
    return array


def build_observations(settings, settings_path):
    """Return deterministic metadata records, without assigning dataset splits."""
    campaign = settings['campaign']
    if not campaign['name'] or Path(campaign['name']).name != campaign['name']:
        raise ValueError('campaign name must be a nonempty path component')
    catalog_path = Path(settings_path).resolve().parent / campaign['profile_catalog']
    catalog = yaml.safe_load(catalog_path.read_text())
    heights = _values(catalog['heights_m'], 'heights', 0, np.inf)
    speed = _values(catalog['wind_speed_m_s'], 'wind speeds', 0, np.inf)
    direction = _values(catalog['wind_direction_deg'], 'wind directions', 0, 360)
    if len(speed) != len(heights) or len(direction) != len(heights):
        raise ValueError('Heights, speeds and directions must have matching lengths')
    if np.any(np.diff(heights) <= 0):
        raise ValueError('Layer heights must be strictly increasing')
    wavelength = float(catalog['reference_wavelength_nm'])
    if wavelength != 500:
        raise ValueError('SPECULA atmospheric strength must be referenced to 500 nm')
    outer_scale = float(catalog['outer_scale_m'])
    if not np.isfinite(outer_scale) or outer_scale <= 0:
        raise ValueError('outer_scale_m must be positive and finite')
    profiles = {}
    for name, profile in catalog['profiles'].items():
        fractions = _values(profile['fractions'], 'Cn2 fractions', 0, 1)
        strength = float(profile['integrated_cn2_m_1_3'])
        if (len(fractions) != len(heights) or not np.isclose(fractions.sum(), 1, atol=1e-6)
                or not np.isfinite(strength) or strength <= 0):
            raise ValueError(f'Invalid turbulence profile: {name}')
        profiles[name] = fractions / fractions.sum() * strength
    seed_start = _positive_int(campaign['simulation_seed_start'], 'simulation_seed_start')
    screen = _positive_int(campaign['pixel_phasescreens'], 'pixel_phasescreens')
    if screen & (screen - 1):
        raise ValueError('pixel_phasescreens must be a power of two')
    rng = np.random.default_rng(_positive_int(campaign['sampler_seed'], 'sampler_seed'))
    observations = []

    def append(profile_ids, alpha, zenith, scale, rotation):
        index = len(observations)
        # Reserve a block for any per-layer atmospheric sub-seeds, then CCD seeds.
        seed = seed_start + index * 16
        if len(heights) > 8 or seed + 10 >= 2**32:
            raise ValueError('Seed allocation supports at most eight layers and uint32 seeds')
        strengths = profiles[profile_ids[0]].copy()
        if alpha is not None:
            strengths = (1 - alpha) * strengths + alpha * profiles[profile_ids[1]]
        total = float(strengths.sum())
        r0 = float((.423 * (2 * np.pi / (wavelength * 1e-9))**2 * total)**(-3 / 5))
        observations.append({
            'campaign': campaign['name'], 'observation_index': index,
            'observation_id': f"{campaign['name']}_{index:06d}",
            'source_doi': catalog['source_doi'],
            'profile_kind': campaign['kind'], 'profile_ids': list(profile_ids),
            'interpolation_alpha': None if alpha is None else float(alpha),
            'heights_m': heights.tolist(), 'layer_cn2_m_1_3': strengths.tolist(),
            'integrated_cn2_m_1_3': total, 'cn2_fractions': (strengths / total).tolist(),
            'reference_wavelength_nm': wavelength, 'r0_500nm_m': r0,
            # Match AtmoEvolution's conversion, not the paper's finite-L0 FWHM.
            'seeing_500nm_zenith_arcsec': float(.9759 * .5 / (4.848 * r0)),
            'outer_scale_m': outer_scale, 'zenith_angle_deg': float(zenith),
            'wind_speed_scale': float(scale), 'wind_rotation_deg': float(rotation),
            'wind_speed_m_s': (speed * scale).tolist(),
            'wind_direction_deg': ((direction + rotation) % 360).tolist(),
            'wind_model': 'MAOS defaults; artificial 50-degree shear plus global rotation',
            'pixel_phasescreens': screen, 'sampler_seed': campaign['sampler_seed'],
            'science_time_s': float(settings['science_time']),
            'settling_time_s': float(settings['settling_time']),
            'random_seeds': {'atmosphere': seed, 'photon': seed + 8,
                             'readout': seed + 9, 'excess': seed + 10},
        })

    if campaign['kind'] == 'reference':
        count = _positive_int(campaign['realizations'], 'realizations')
        angles = _values(campaign['zenith_angles_deg'], 'zenith angles', 0, 89)
        scales = _values(campaign['wind_speed_scales'], 'speed scales', 0, np.inf,
                         strictly_positive=True)
        for profile in campaign['profiles']:
            for zenith in angles:
                for scale in scales:
                    rotations = 360 * (np.arange(count) + rng.random(count)) / count
                    rng.shuffle(rotations)
                    for rotation in rotations:
                        append([profile], None, zenith, scale, rotation)
    elif campaign['kind'] == 'intermediate':
        count = _positive_int(campaign['observations_per_pair'], 'observations_per_pair')
        angles = _range(campaign['zenith_angle_range_deg'], 'zenith range', 0, 89)
        scales = _range(campaign['wind_speed_scale_range'], 'speed scale range', 0, np.inf,
                        strictly_positive=True)
        for pair in campaign['profile_pairs']:
            if len(pair) != 2 or pair[0] == pair[1]:
                raise ValueError('Each interpolation pair must name two distinct profiles')
            # Midpoints of strata exclude endpoints even if the RNG returns zero.
            alphas = (np.arange(count) + .5) / count
            rng.shuffle(alphas)
            for alpha in alphas:
                append(pair, alpha, rng.uniform(*angles), rng.uniform(*scales), rng.uniform(0, 360))
    else:
        raise ValueError(f"Unknown campaign kind: {campaign['kind']}")
    if not observations:
        raise ValueError('Campaign must contain at least one observation')
    return observations


def apply_observation(params, settings, observation):
    """Apply one record to the existing science graph, preserving its AO timing."""
    if settings['science_time'] != 10 or settings['long_exposure'] != 10:
        raise ValueError('Profile datasets require 10 seconds of science and long exposure')
    if params['main']['time_step'] != .002 or settings['short_frames'] != 100:
        raise ValueError('Profile datasets require 500 Hz and 100-frame samples')
    params['main']['zenithAngleInDeg'] = observation['zenith_angle_deg']
    params['seeing']['constant'] = observation['seeing_500nm_zenith_arcsec']
    params['wind_speed']['constant'] = observation['wind_speed_m_s']
    params['wind_direction']['constant'] = observation['wind_direction_deg']
    params['atmo'].update(heights=observation['heights_m'], Cn2=observation['cn2_fractions'],
                          L0=observation['outer_scale_m'],
                          pixel_phasescreens=observation['pixel_phasescreens'],
                          seed=observation['random_seeds']['atmosphere'])
    params['detector_truth_sh'].update({f'{name}_seed': observation['random_seeds'][name]
                                      for name in ('photon', 'readout', 'excess')})
    # Conservative diagonal field footprint, plus a full sampling-step margin.
    main = params['main']
    airmass = 1 / np.cos(np.radians(observation['zenith_angle_deg']))
    footprint = (main['pixel_pupil'] * main['pixel_pitch']
                 + max(observation['heights_m']) * airmass
                 * np.radians(settings['field_width_arcsec'] * np.sqrt(2) / 3600))
    duration = settings['settling_time'] + settings['science_time'] + main['time_step']
    required = max(observation['wind_speed_m_s']) * duration + footprint + 2 * main['pixel_pitch']
    if observation['pixel_phasescreens'] * main['pixel_pitch'] <= required:
        raise ValueError('Phase screens are too short: increase pixel_phasescreens to avoid cycling')
