'''
The tutorials that use only iteras objects run without SPECULA: their code,
extracted from the documentation, runs in a process where importing specula fails.
'''
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TUTORIALS = REPO_ROOT / 'docs' / 'tutorials'


def _code_blocks(rst_file, language):
    '''The code blocks of *language* in an rst file, without their indentation'''
    blocks = []
    lines = (TUTORIALS / rst_file).read_text(encoding='utf-8').splitlines()
    i = 0
    while i < len(lines):
        if re.match(rf'\s*\.\. code-block:: {language}\s*$', lines[i]):
            i += 1
            block = []
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith((' ', '\t'))):
                block.append(lines[i])
                i += 1
            blocks.append(textwrap.dedent('\n'.join(block)))
        else:
            i += 1
    return blocks


def _run_without_specula(code, cwd):
    script = 'import sys\nsys.modules["specula"] = None   # any import of specula fails\n' + code
    # The repository first, since the code runs in another directory (and iteras
    # might also be installed from another checkout)
    env = dict(os.environ, MPLBACKEND='Agg',
               PYTHONPATH=os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get('PYTHONPATH')])))
    return subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                          cwd=cwd, env=env)


class TestIterasTutorials(unittest.TestCase):

    def test_step_response_tutorial(self):
        params = yaml.safe_load(_code_blocks('step_response_tutorial.rst', 'yaml')[0])
        self.assertEqual(params['main']['class'], 'SimParams')
        # Without the display, shorter, and storing the actuator output
        del params['all_disp']
        params['main']['total_time'] = 0.05
        with tempfile.TemporaryDirectory() as tmp:
            params['store'] = {'class': 'DataStore', 'store_dir': tmp, 'create_tn': False,
                               'data_format': 'npz',
                               'inputs': {'input_list': ['out-lowpass.out_comm']}}
            yml = os.path.join(tmp, 'params.yml')
            with open(yml, 'w') as f:
                yaml.dump(params, f)
            r = _run_without_specula(textwrap.dedent(f'''
                import numpy as np
                import iteras
                iteras.init(-1)
                assert iteras.__file__.startswith({str(REPO_ROOT)!r}), iteras.__file__
                from iteras.simul import Simul
                from iteras.lib import storage
                Simul({yml!r}, speed_report=False).run()
                with storage.read_object({os.path.join(tmp, 'out.npz')!r}) as f:
                    out = f.read('data', 0)
                # The actuator follows the square wave of amplitude 5
                assert np.all(np.isfinite(out)) and 3 < np.abs(out).max() < 15, out.ravel()
                '''), cwd=tmp)
            self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])

    def test_control_stability_analysis_tutorial(self):
        try:
            import control  # noqa: F401
        except ImportError:
            self.skipTest('The tutorial needs the control package')
        code = _code_blocks('control_stability_analysis.rst', 'python')[0]
        with tempfile.TemporaryDirectory() as tmp:
            r = _run_without_specula(code, cwd=tmp)
            self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])


if __name__ == '__main__':
    unittest.main()
