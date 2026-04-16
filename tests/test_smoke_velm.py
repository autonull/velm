import os
import subprocess


def test_smoke_train():
    # run the train script for a few steps
    script_path = "experiments/train_velm_full.py"
    if not os.path.exists(script_path):
        script_path = "experiments/train_velm_full_proper.py"

    p = subprocess.run(["python3", script_path, "--steps", "20", "--out", "outputs_full_smoke"], check=True)
    assert os.path.exists('outputs_full_smoke/smoke.txt')
