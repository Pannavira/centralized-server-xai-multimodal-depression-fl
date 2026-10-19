import pandas as pd
import json
import hashlib
from pathlib import Path

run_dir = Path("storage/federated/runs/FL-RUN-20261016-130403-45D587")

print("=== DATASET MANIFEST ===")
with open(run_dir / "dataset_manifest.json") as f:
    ds_m = json.load(f)
print("Zero leakage verified:", ds_m["zero_leakage_verified"])
print(f"Total Train Participants: {ds_m.get('total_train_participants')}")
print(f"Total Val Participants  : {ds_m.get('total_val_participants')}")
print(f"Total Test Participants : {ds_m.get('total_test_participants')}")
print("Train Partitions:")
for k, v in ds_m.get("client_train_partitions", {}).items():
    print(f"  {k}: samples={v.get('samples')}, sha256={v.get('sha256')[:16]}")
print("Val Partitions:")
for k, v in ds_m.get("client_val_partitions", {}).items():
    print(f"  {k}: samples={v.get('samples')}, sha256={v.get('sha256')[:16]}")

print("\n=== METRICS TRAJECTORY ===")
df_m = pd.read_csv(run_dir / "metrics.csv")
print(f"Total metrics rows: {len(df_m)}")
sample_rounds = [0, 1, 5, 10, 12, 18, 20, 22, 30, 40, 43, 48, 50]
for r in sample_rounds:
    sub = df_m[df_m["round"] == r]
    if len(sub) > 0:
        row = sub.iloc[0]
        print(f"Round {r:02d} | Loss: {row['test_loss']:.4f} | MAE: {row['mae']:.4f} | RMSE: {row['rmse']:.4f} | F1: {row['macro_f1']:.4f} | Rec: {row['recall']:.4f} | Spec: {row['specificity']:.4f} | AUC: {row['auc_roc']}")

best_mae_idx = df_m["mae"].idxmin()
best_mae_row = df_m.loc[best_mae_idx]
best_f1_idx = df_m["macro_f1"].idxmax()
best_f1_row = df_m.loc[best_f1_idx]
final_row = df_m.iloc[-1]

print(f"\nBest Test MAE (reporting): Round {int(best_mae_row['round'])} | MAE: {best_mae_row['mae']:.4f} | F1: {best_mae_row['macro_f1']:.4f}")
print(f"Best Test F1  (reporting): Round {int(best_f1_row['round'])} | F1: {best_f1_row['macro_f1']:.4f} | MAE: {best_f1_row['mae']:.4f}")
print(f"Final Round 50          : MAE: {final_row['mae']:.4f} | RMSE: {final_row['rmse']:.4f} | F1: {final_row['macro_f1']:.4f} | Rec: {final_row['recall']:.4f} | Spec: {final_row['specificity']:.4f} | AUC: {final_row['auc_roc']}")

print("\n=== CLIENT METRICS TRAJECTORY ===")
df_cl = pd.read_csv(run_dir / "client_metrics.csv")
print(f"Total client metric entries: {len(df_cl)}")
print(f"Unique clients: {df_cl['client_id'].unique()}")
for r in [1, 10, 25, 50]:
    sub = df_cl[df_cl["round"] == r]
    avg_train_loss = sub["train_loss"].mean()
    avg_train_mae = sub["train_mae"].mean()
    avg_val_loss = sub["val_loss"].mean()
    avg_val_mae = sub["val_mae"].mean()
    print(f"Round {r:02d} Client Avg -> Train Loss: {avg_train_loss:.4f}, Train MAE: {avg_train_mae:.4f}, Val Loss: {avg_val_loss:.4f}, Val MAE: {avg_val_mae:.4f}")

print("\n=== COMMUNICATION PAYLOAD ===")
df_c = pd.read_csv(run_dir / "communication.csv")
r0 = df_c.iloc[0]
r1 = df_c.iloc[1]
r50 = df_c.iloc[-1]
print(f"Payload per round: model={r1['model_parameter_mb']:.4f} MB | downlink={r1['estimated_downlink_mb']:.4f} MB | uplink={r1['estimated_uplink_mb']:.4f} MB | total_round={r1['estimated_total_mb']:.4f} MB")
print(f"Cumulative payload Round 50: {r50['cumulative_mb']:.4f} MB ({r50['cumulative_parameter_bytes'] / (1024**3):.4f} GB)")

print("\n=== CHECKPOINTS SHA-256 ===")
def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

r0_sha = get_sha256(run_dir / "rounds/round_000/model.pt")
r50_sha = get_sha256(run_dir / "rounds/round_050/model.pt")
ref_sha = get_sha256("storage/models/fl_iid/model.pt")
print(f"Round 000 Checkpoint SHA256: {r0_sha}")
print(f"Round 050 Checkpoint SHA256: {r50_sha}")
print(f"Historical Reference SHA256 : {ref_sha}")

print("\n=== REFERENCE COMPARISON JSON ===")
with open(run_dir / "reference_comparison.json") as f:
    ref_comp = json.load(f)
print(f"Prediction Correlation: {ref_comp.get('prediction_correlation')}")
print(f"Cohort Mean Abs Diff : {ref_comp.get('cohort_mean_absolute_difference')}")
print("Metrics Comparison:")
for k, v in ref_comp.get("metrics_comparison", {}).items():
    print(f"  {k:12s} | Historical: {v.get('historical'):.4f} | Reproduction: {v.get('reproduction'):.4f} | Delta: {v.get('absolute_delta'):.4f}")

print("\n=== MANIFEST ===")
with open(run_dir / "manifest.json") as f:
    man = json.load(f)
print(f"Run ID: {man.get('run_id')}")
print(f"Completed At: {man.get('completed_at')}")
print(f"Scientific Status: {man.get('scientific_status')}")
print(f"Total Checkpoints in Manifest: {len(man.get('round_checkpoints', {}))}")
