"""
scripts/prepare_seed_initializations.py
=======================================
Generates, freezes, and registers authoritative Round-000 model checkpoints
for all target seeds in FED-MIND Milestone 5.6: [42, 43, 44, 45, 46].
"""

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from app.core.model import MultimodalBaselineModel
from federated.evaluation import GlobalEvaluator
from federated.server_app import get_environment_info
from federated.task import build_model


def get_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 70)
    print("PREPARING MULTI-SEED INITIALIZATIONS (M5.6)")
    print("=" * 70)

    base_init_dir = Path("storage/federated/multiseed/initializations")
    base_init_dir.mkdir(parents=True, exist_ok=True)

    seeds = [42, 43, 44, 45, 46]
    hist_r0_ckpt = Path("storage/federated/runs/FL-RUN-20261016-130403-45D587/rounds/round_000/model.pt")

    evaluator = GlobalEvaluator(device="cpu")
    env_info = get_environment_info()

    initializations_summary = {}

    for s in seeds:
        seed_dir = base_init_dir / f"seed_{s}" / "round_000"
        seed_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = seed_dir / "model.pt"
        manifest_path = seed_dir / "manifest.json"

        if s == 42:
            print(f"\n[Seed 42] Copying verified historical M5.1 checkpoint...")
            if not hist_r0_ckpt.exists():
                raise FileNotFoundError(f"Historical checkpoint not found: {hist_r0_ckpt}")
            shutil.copy2(hist_r0_ckpt, ckpt_path)
            sha = get_file_sha256(ckpt_path)
            assert sha == "a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973", (
                f"Historical SHA mismatch: {sha}"
            )
            print(f"  Verified SHA-256: {sha}")
        else:
            print(f"\n[Seed {s}] Generating fresh deterministic model weights...")
            model = build_model(seed=s)
            torch.save(model.state_dict(), ckpt_path)
            sha = get_file_sha256(ckpt_path)
            print(f"  Generated SHA-256: {sha}")

        # Strict load verification
        test_model = build_model()
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        test_model.load_state_dict(sd, strict=True)
        param_count = sum(p.numel() for p in test_model.parameters())

        # Canonical baseline evaluation
        metrics = evaluator.evaluate_state_dict(sd)
        print(f"  R0 Eval: Loss={metrics.get('test_loss')}, MAE={metrics.get('mae')}, RMSE={metrics.get('rmse')}, F1={metrics.get('macro_f1')}")

        manifest = {
            "seed": s,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_architecture": "MultimodalBaselineModel",
            "model_parameter_count": param_count,
            "checkpoint_sha256": sha,
            "checkpoint_byte_size": ckpt_path.stat().st_size,
            "initialization_source": "historical_m5_1" if s == 42 else f"build_model(seed={s})",
            "initial_evaluation_metrics": metrics,
            "environment_info": env_info,
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        initializations_summary[f"seed_{s}"] = {
            "seed": s,
            "checkpoint_path": str(ckpt_path).replace("\\", "/"),
            "checkpoint_sha256": sha,
            "parameter_count": param_count,
            "r0_mae": metrics.get("mae"),
            "r0_rmse": metrics.get("rmse"),
            "r0_f1": metrics.get("macro_f1"),
            "r0_recall": metrics.get("recall"),
            "r0_specificity": metrics.get("specificity"),
        }

    # Consolidated manifest in storage/federated/multiseed/m5_6/
    m5_6_dir = Path("storage/federated/multiseed/m5_6")
    m5_6_dir.mkdir(parents=True, exist_ok=True)
    summary_path = m5_6_dir / "seed_initializations.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(initializations_summary, f, indent=2)

    print(f"\n[DONE] Initializations saved to {base_init_dir}")
    print(f"[DONE] Consolidated summary saved to {summary_path}")


if __name__ == "__main__":
    main()
