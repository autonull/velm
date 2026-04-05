import subprocess
import os


def test_train_velm_smoke():
    out = 'outputs/test_smoke'
    if os.path.exists(out):
        # ensure clean
        pass
    cmd = ['python3', 'src/train_velm_full.py', '--steps', '1', '--out', out, '--device', 'cpu']
    res = subprocess.run(cmd, check=True)
    # smoke artifact
    assert os.path.exists(os.path.join(out, 'smoke.txt'))
