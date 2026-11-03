"""Check paper profiles, independent realizations and per-observation routing."""

from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml


CONFIG = Path(__file__).resolve().parents[1] / 'config/START-SCAO'
module = importlib.util.spec_from_file_location('campaign_workflow', CONFIG / 'run_psf_grid.py')
workflow = importlib.util.module_from_spec(module)
module.loader.exec_module(workflow)


def campaign(name):
    path = CONFIG / f'{name}_profile_dataset.yml'
    settings = yaml.safe_load(path.read_text())
    return path, settings, workflow.build_observations(settings, path)


def test_reference_cases_match_paper_and_have_independent_unsplit_realizations():
    path, settings, observations = campaign('reference')
    assert len(observations) == 450
    assert observations == workflow.build_observations(settings, path)
    counts = Counter((o['profile_ids'][0], o['zenith_angle_deg'], o['wind_speed_scale'])
                     for o in observations)
    assert len(counts) == 45 and set(counts.values()) == {10}
    expected = {'good': (3.5781e-13, .149, .676),
                'moderate': (5.2770e-13, .118, .853),
                'bad': (8.1360e-13, .091, 1.106)}
    for observation in observations:
        strength, r0, seeing = expected[observation['profile_ids'][0]]
        assert observation['integrated_cn2_m_1_3'] == pytest.approx(strength, abs=1e-20)
        assert observation['r0_500nm_m'] == pytest.approx(r0, abs=1e-4)
        assert observation['seeing_500nm_zenith_arcsec'] == pytest.approx(seeing, abs=.001)
        assert sum(observation['cn2_fractions']) == pytest.approx(1)
        assert observation['outer_scale_m'] == 30
        assert observation['heights_m'][0] == 0
        assert 'split' not in observation
    for start in range(0, 450, 10):
        strata = [int(o['wind_rotation_deg'] // 36) for o in observations[start:start + 10]]
        assert sorted(strata) == list(range(10))


def test_interpolation_uses_absolute_strengths_and_no_seed_overlap():
    path, settings, observations = campaign('intermediate')
    assert len(observations) == 150
    assert observations == workflow.build_observations(settings, path)
    assert Counter(tuple(o['profile_ids']) for o in observations) == {
        ('good', 'moderate'): 75, ('moderate', 'bad'): 75}
    catalog = yaml.safe_load((CONFIG / 'oya2014_profiles.yml').read_text())['profiles']
    for o in observations:
        a, b = [catalog[name] for name in o['profile_ids']]
        alpha = o['interpolation_alpha']
        assert 0 < alpha < 1
        expected = ((1 - alpha) * a['integrated_cn2_m_1_3'] * np.array(a['fractions'])
                    + alpha * b['integrated_cn2_m_1_3'] * np.array(b['fractions']))
        np.testing.assert_allclose(o['layer_cn2_m_1_3'], expected, rtol=1e-14, atol=0)
        np.testing.assert_allclose(o['cn2_fractions'], expected / expected.sum())
        assert 0 <= o['zenith_angle_deg'] <= 60
        assert .5 <= o['wind_speed_scale'] <= 2
        assert 'split' not in o
    all_observations = campaign('reference')[2] + observations
    seeds = [value for o in all_observations for value in o['random_seeds'].values()]
    assert len(seeds) == len(set(seeds))
    assert len({o['observation_id'] for o in all_observations}) == 600


def test_all_observations_preserve_timing_and_fit_without_screen_cycling():
    base = yaml.safe_load((CONFIG / 'params_truthwfs_scao.yml').read_text())
    for kind in ('reference', 'intermediate'):
        _, settings, observations = campaign(kind)
        for o in observations:
            params = deepcopy(base)
            workflow.apply_observation(params, settings, o)
            short, long = workflow.validate_settings(settings, params)
            assert (short, long) == (100, 5000)
            assert settings['science_time'] / params['main']['time_step'] / short == 50
            assert params['dm'] == base['dm']
            assert params['integrator'] == base['integrator']
            assert params['detector_truth_sh']['photon_seed'] == o['random_seeds']['photon']
            assert params['atmo']['Cn2'] == o['cn2_fractions']
        fast = deepcopy(max(observations, key=lambda o: max(o['wind_speed_m_s'])))
        fast['pixel_phasescreens'] = 8192
        with pytest.raises(ValueError, match='too short'):
            workflow.apply_observation(deepcopy(base), settings, fast)
        with pytest.raises(ValueError, match='10 seconds'):
            workflow.apply_observation(deepcopy(base), {**settings, 'science_time': 20}, observations[0])


def test_cli_manifest_and_observation_routing(tmp_path, monkeypatch):
    path, _, observations = campaign('reference')
    result = subprocess.run([sys.executable, str(CONFIG / 'run_psf_grid.py'),
                             'manifest', '--settings', str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert [json.loads(line) for line in result.stdout.splitlines()] == observations
    seen = []

    def capture(params, path):
        seen.append(params)
        path.write_text(yaml.safe_dump(params))

    monkeypatch.setattr(workflow, 'run_params', capture)
    run_dir = tmp_path / 'observation'
    args = SimpleNamespace(settings=path, run_dir=run_dir, science_time=None, observation=449,
                           displays=False, shard_size=4, max_shard_gib=1.)
    workflow.run_science(args)
    assert len(seen) == 1
    assert seen[0]['main']['total_time'] == 11
    assert seen[0]['dataset']['start_time'] == 1
    assert seen[0]['dataset']['observation_metadata'] == observations[449]
    assert json.loads((run_dir / 'observation.json').read_text()) == observations[449]
    assert 'campaign' not in yaml.safe_load((run_dir / 'psf_grid.yml').read_text())
    for index in (None, -1, 450):
        args.observation = index
        with pytest.raises(ValueError, match='Choose --observation'):
            workflow.run_science(args)
    assert len(seen) == 1
