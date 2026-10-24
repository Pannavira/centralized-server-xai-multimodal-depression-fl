import hashlib
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from federated.evaluation import GlobalEvaluator
from app.core.model import MultimodalBaselineModel

runs = {
    "M5.1_IID": "FL-RUN-20261016-130403-45D587",
    "M5.2_Label": "FL-LABEL-SKEW-20261016-141108-7865D4",
    "M5.3_Quantity": "FL-QUANTITY-SKEW-20261019-110746-3A4BFF",
    "M5.4_Modality": "FL-MODALITY-SKEW-20261019-142242-CA6D11",
    "M5.5_Combined": "FL-COMBINED-NONIID-20261019-154311-CCF7B4",
}

base_dir = Path("storage/federated/runs")


def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 70)
    print("AUDIT: ROUND 0 AND HISTORICAL RUNS")
    print("=" * 70)

    evaluator = GlobalEvaluator(device="cpu")
    print(f"GlobalEvaluator initialized on CPU. Test samples: {len(evaluator.test_loader.dataset)}")

    results = []

    for name, run_id in runs.items():
        run_path = base_dir / run_id
        r0_ckpt = run_path / "rounds" / "round_000" / "model.pt"
        r50_ckpt = run_path / "rounds" / "round_050" / "model.pt"
        metrics_csv = run_path / "metrics.csv"
        manifest_json = run_path / "manifest.json"

        print(f"\n--- {name} ({run_id}) ---")
        if not r0_ckpt.exists():
            print(f"ERROR: {r0_ckpt} not found!")
            continue

        r0_sha = get_sha256(r0_ckpt)
        r50_sha = get_sha256(r50_ckpt) if r50_ckpt.exists() else "NOT_FOUND"

        print(f"R0 SHA256 : {r0_sha}")
        print(f"R50 SHA256: {r50_sha}")

        # Metrics CSV inspection
        df = pd.read_csv(metrics_csv)
        r0_csv = df[df["round"] == 0].iloc[0].to_dict()
        r50_csv = df[df["round"] == 50].iloc[0].to_dict()

        print(
            f"CSV R0 : Loss={r0_csv.get('test_loss'):.4f}, MAE={r0_csv.get('mae'):.4f}, RMSE={r0_csv.get('rmse'):.4f}, F1={r0_csv.get('macro_f1'):.4f}, Rec={r0_csv.get('recall'):.4f}"
        )
        print(
            f"CSV R50: Loss={r50_csv.get('test_loss'):.4f}, MAE={r50_csv.get('mae'):.4f}, RMSE={r50_csv.get('rmse'):.4f}, F1={r50_csv.get('macro_f1'):.4f}, Rec={r50_csv.get('recall'):.4f}"
        )

        # Canonical Re-evaluation of R0
        r0_state = torch.load(r0_ckpt, map_location="cpu", weights_only=True)
        if isinstance(r0_state, dict) and "state_dict" in r0_state:
            r0_state = r0_state["state_dict"]
        eval_r0 = evaluator.evaluate_state_dict(r0_state)
        print(
            f"EVAL R0: Loss={eval_r0.get('test_loss')}, MAE={eval_r0.get('mae')}, RMSE={eval_r0.get('rmse')}, F1={eval_r0.get('macro_f1')}, Rec={eval_r0.get('recall')}"
        )

        # Canonical Re-evaluation of R50
        r50_state = torch.load(r50_ckpt, map_location="cpu", weights_only=True)
        if isinstance(r50_state, dict) and "state_dict" in r50_state:
            r50_state = r50_state["state_dict"]
        eval_r50 = evaluator.evaluate_state_dict(r50_state)
        print(
            f"EVAL R50: Loss={eval_r50.get('test_loss')}, MAE={eval_r50.get('mae')}, RMSE={eval_r50.get('rmse')}, F1={eval_r50.get('macro_f1')}, Rec={eval_r50.get('recall')}"
        )

        results.append(
            {
                "scenario": name,
                "run_id": run_id,
                "r0_sha": r0_sha,
                "r50_sha": r50_sha,
                "r0_csv_mae": r0_csv.get("mae"),
                "r0_eval_mae": eval_r0.get("mae"),
                "r50_csv_mae": r50_csv.get("mae"),
                "r50_eval_mae": eval_r50.get("mae"),
                "r0_eval_f1": eval_r0.get("macro_f1"),
                "r50_eval_f1": eval_r50.get("macro_f1"),
            }
        )

    print("\n" + "=" * 70)
    print("SUMMARY AUDIT TABLE:")
    df_res = pd.DataFrame(results)
    print(df_res.to_string())


if __name__ == "__main__":
    main()
