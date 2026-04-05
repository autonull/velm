import subprocess
import os


def test_train_velm_smoke():
    out = 'outputs/test_smoke'
    if os.path.exists(out):
        # ensure clean
        pass

    script_path = 'experiments/train_velm_full.py'
    if not os.path.exists(script_path):
        script_path = 'experiments/train_velm_full_proper.py'

    cmd = ['python3', script_path, '--steps', '1', '--out', out, '--device', 'cpu']
    res = subprocess.run(cmd, check=True)
    # smoke artifact
    assert os.path.exists(os.path.join(out, 'smoke.txt'))
