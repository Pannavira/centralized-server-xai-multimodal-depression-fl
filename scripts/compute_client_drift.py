"""
scripts/compute_client_drift.py
================================
Menghitung client update L2 norm secara deterministik untuk setiap ronde dan klien,
memasukkannya ke client_metrics.csv, dan memicu pembuatan plots/client_drift_label_skew.png.
"""

import math
from pathlib import Path
import shutil
import sys

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import torch

from federated.data import build_client_dataloaders
from federated.task import build_model, train_local
from federated.comparison import generate_iid_vs_label_skew_comparison

RUN_DIR = Path("storage/federated/runs/FL-LABEL-SKEW-20261016-141108-7865D4")
PARTITION_DIR = Path("storage/federated/partitions/label_skew_seed42")

client_node_to_partition = {
    "node_8380547306511574968": 0,
    "node_11589894290611338934": 1,
    "node_13750544803340716973": 2,
}

def main():
    csv_path = RUN_DIR / "client_metrics.csv"
    df = pd.read_csv(csv_path)
    
    print(f"Memproses {len(df)} baris pada client_metrics.csv...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Siapkan dataloader untuk masing-masing client
    loaders = {}
    for node_id, part_id in client_node_to_partition.items():
        tr_loader, val_loader, _, _ = build_client_dataloaders(
            partition_id=part_id,
            scenario="non_iid_label_skew",
            batch_size=16,
        )
        loaders[node_id] = tr_loader

    drift_results = []
    
    # Kelompokkan per ronde
    for r in range(1, 51):
        prev_round = r - 1
        prev_ckpt = RUN_DIR / "rounds" / f"round_{prev_round:03d}" / "model.pt"
        if not prev_ckpt.exists():
            print(f"Checkpoint round_{prev_round:03d} tidak ditemukan, skip.")
            continue
            
        base_state = torch.load(prev_ckpt, map_location="cpu", weights_only=False)
        
        # Sub-dataframe untuk ronde r
        sub_df = df[df["round"] == r]
        
        for idx, row in sub_df.iterrows():
            node_id = str(row["client_id"])
            tr_loader = loaders.get(node_id)
            if tr_loader is None:
                continue
                
            model = build_model()
            model.load_state_dict(base_state, strict=True)
            
            # Local training 2 epoch sama persis dengan konfigurasi FL
            train_local(
                model=model,
                train_loader=tr_loader,
                epochs=2,
                lr=1e-4,
                weight_decay=0.01,
                device=device,
            )
            
            # Hitung L2 distance terhadap base_state
            local_sd = model.state_dict()
            drift_sq = 0.0
            for k, v_glob in base_state.items():
                if k in local_sd:
                    diff = (local_sd[k].cpu() - v_glob.cpu()).float()
                    drift_sq += float(torch.sum(diff ** 2).item())
            drift_l2 = float(math.sqrt(drift_sq))
            
            df.loc[idx, "client_update_l2_norm"] = round(drift_l2, 6)
            
        if r % 10 == 0 or r == 1:
            print(f"Ronde {r}/50 selesai dihitung drift-nya.")

    # Simpan kembali df ke client_metrics.csv
    df.to_csv(csv_path, index=False)
    print("Berhasil memperbarui client_metrics.csv dengan client_update_l2_norm!")
    
    # Jalankan komparasi dan plot
    generate_iid_vs_label_skew_comparison(label_skew_run_dir=RUN_DIR)
    print("Semua grafik komparatif termasuk client_drift_label_skew.png berhasil dihasilkan!")

if __name__ == "__main__":
    main()
