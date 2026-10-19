"""
scripts/design_label_skew.py
============================
Mengkaji dan memvalidasi partisi controlled_label_skew_v1 untuk Milestone 5.2.
Invarian:
1. Pool subjek identik 100% dengan M5.1 (163 subjek).
2. Kuantitas klien: 54, 55, 54.
3. Total Positif = 50, Total Negatif = 113.
4. Zero duplicate, zero leakage.
5. Maximize Label Skew difference between Client 1 (Depression-Heavy) and Client 2 (Control-Heavy),
   dengan Client 3 sebagai mixed/remainder.
"""

from pathlib import Path
import pandas as pd
import numpy as np

iid_dir = Path("C:/Wira/Kuliah/Penelitian/clinical-xai-multimodal-depression-fl/federated_learning_output/IID_3Clients")
dfs = [pd.read_csv(iid_dir / f"client_{c}_train.csv") for c in [1, 2, 3]]
all_train = pd.concat(dfs, ignore_index=True)

pos_df = all_train[all_train["phq8_score"] >= 10.0].copy()
neg_df = all_train[all_train["phq8_score"] < 10.0].copy()

total_pos = len(pos_df) # 50
total_neg = len(neg_df) # 113

print(f"Total training pool: {len(all_train)} (Positive: {total_pos}, Negative: {total_neg})")

# Evaluation of candidate allocations
# Target: Client 1 ~ 80% (43.2), Client 2 ~ 20% (11.0), Client 3 remainder.
# Total positive = 50.
# Since 43 + 11 = 54 > 50, exactly (43, 11) is mathematically impossible.
# Let's see nearest feasible allocations:

results = []
for p1 in range(1, 51):
    for p2 in range(1, 51):
        p3 = total_pos - p1 - p2
        if p3 < 1:
            continue
        n1 = 54 - p1
        n2 = 55 - p2
        n3 = 54 - p3
        if n1 < 1 or n2 < 1 or n3 < 1:
            continue
        
        pr1 = p1 / 54.0
        pr2 = p2 / 55.0
        pr3 = p3 / 54.0
        
        dist_abs = abs(pr1 - 0.80) + abs(pr2 - 0.20)
        skew_diff = pr1 - pr2
        
        results.append({
            "p1": p1, "p2": p2, "p3": p3,
            "n1": n1, "n2": n2, "n3": n3,
            "pr1": pr1, "pr2": pr2, "pr3": pr3,
            "dist_abs": dist_abs,
            "skew_diff": skew_diff
        })

print("\n--- TOP CANDIDATES MINIMIZING DISTANCE TO TARGET (80%, 20%) ---")
results_target = sorted(results, key=lambda x: (x["dist_abs"], -x["skew_diff"]))
for r in results_target[:8]:
    print(f"p1={r['p1']:2d} ({r['pr1']*100:5.1f}%), p2={r['p2']:2d} ({r['pr2']*100:5.1f}%), p3={r['p3']:2d} ({r['pr3']*100:5.1f}%) | "
          f"dist={r['dist_abs']:.4f}, skew_diff={r['skew_diff']*100:5.1f}%")

print("\n--- TOP CANDIDATES WITH MIXED REMAINDER C3 (6 <= p3 <= 15) ---")
results_mixed = [r for r in results if 6 <= r["p3"] <= 15 and r["p1"] >= 30 and r["p2"] <= 10]
results_mixed.sort(key=lambda x: -x["skew_diff"])
for r in results_mixed[:10]:
    print(f"p1={r['p1']:2d} ({r['pr1']*100:5.1f}%), p2={r['p2']:2d} ({r['pr2']*100:5.1f}%), p3={r['p3']:2d} ({r['pr3']*100:5.1f}%) | "
          f"dist={r['dist_abs']:.4f}, skew_diff={r['skew_diff']*100:5.1f}%")

