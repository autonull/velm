import os
import subprocess


def test_smoke_train():
    # run the train script for a few steps
    p = subprocess.run(["python3", "src/train_velm_full.py", "--steps", "20", "--out", "outputs_full_smoke"], check=True)
    assert os.path.exists('outputs_full_smoke/smoke.txt')
