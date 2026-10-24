"""
scripts/test_gate_b.py
Integration test for Gate B: run a 2-round validation on seed 43 for the 5 scenarios.
"""
import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

# Add .venv Scripts to PATH
venv_scripts = r"c:\Wira\Kuliah\Penelitian\clinical-xai-multimodal-depression-fl\.venv\Scripts"
if venv_scripts not in os.environ.get("PATH", ""):
    os.environ["PATH"] = venv_scripts + os.pathsep + os.environ.get("PATH", "")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

SCENARIOS = [
    ("IID_3Clients", "storage/federated/multiseed/partitions/seed_43/IID_3Clients"),
    ("label_skew", "storage/federated/multiseed/partitions/seed_43/label_skew"),
    ("quantity_skew", "storage/federated/multiseed/partitions/seed_43/quantity_skew"),
    ("modality_skew", "storage/federated/multiseed/partitions/seed_43/modality_skew"),
    ("combined_skew", "storage/federated/multiseed/partitions/seed_43/combined_skew"),
]

def run_integration(scenario_name, part_dir, rounds=2):
    print(f"\n=======================================================")
    print(f"Testing Scenario: {scenario_name} (Rounds: {rounds})")
    print(f"=======================================================")
    
    cfg_str = (
        f'num-server-rounds={rounds} '
        f'seed=43 '
        f'scenario="{scenario_name}" '
        f'partition-dir="{part_dir}"'
    )
    
    cmd = ["flwr", "run", ".", "--stream", "--run-config", cfg_str]
    print(f"Command: {' '.join(cmd)}")
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT tail:")
    print(res.stdout[-1500:] if len(res.stdout) > 1500 else res.stdout)
    if res.returncode != 0:
        print("STDERR:")
        print(res.stderr[-2000:] if len(res.stderr) > 2000 else res.stderr)
        raise RuntimeError(f"Run failed for {scenario_name} with return code {res.returncode}")
        
    print(f"[SUCCESS] {scenario_name} completed cleanly.")

if __name__ == "__main__":
    # Test just the first one first: IID_3Clients
    run_integration(SCENARIOS[0][0], SCENARIOS[0][1], rounds=2)
