"""
tests/test_federated_data.py
============================
Pengujian Unit untuk Adapter Dataset dan Verifikasi Privasi Zero-Leakage (Milestone 4).
"""

from pathlib import Path
import pandas as pd
import pytest
import torch

from federated.data import (
    DEFAULT_PARTITION_DIR,
    MultimodalPartitionDataset,
    build_client_dataloaders,
    build_global_test_dataloader,
    get_client_partition_paths,
    verify_zero_leakage,
)


def test_client_partitions_resolve():
    """Memverifikasi bahwa tepat 3 partisi klien dapat ditemukan dan dimuat."""
    for pid in [0, 1, 2]:
        train_path, val_path = get_client_partition_paths(partition_id=pid)
        assert train_path.exists(), f"Train CSV tidak ditemukan untuk partition {pid}: {train_path}"
        assert val_path.exists(), f"Val CSV tidak ditemukan untuk partition {pid}: {val_path}"

        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        assert len(df_train) > 0, f"Partition {pid} train kosong"
        assert len(df_val) > 0, f"Partition {pid} val kosong"


def test_zero_leakage_assertion():
    """
    Memverifikasi aksioma privasi: Intersection(all_client_train_ids, global_test_ids) == empty set.
    """
    test_path = DEFAULT_PARTITION_DIR / "global_test_index.csv"
    assert test_path.exists(), f"File global_test_index.csv tidak ditemukan di {test_path}"

    test_df = pd.read_csv(test_path)
    client_dfs = []
    for pid in [0, 1, 2]:
        train_path, _ = get_client_partition_paths(partition_id=pid)
        client_dfs.append(pd.read_csv(train_path))

    # 1. Pastikan zero leakage pada data asli
    passed, leakage = verify_zero_leakage(client_dfs, test_df)
    assert passed is True
    assert len(leakage) == 0

    # 2. Pastikan validator melempar ValueError jika sengaja disuntikkan data bocor
    leak_df = client_dfs[0].copy()
    leak_df.iloc[0, leak_df.columns.get_loc("participant_id")] = test_df.iloc[0]["participant_id"]
    with pytest.raises(ValueError, match="DATA LEAKAGE DETECTED"):
        verify_zero_leakage([leak_df], test_df)


def test_dataset_item_shapes_and_types():
    """Memverifikasi bentuk (shape) dan tipe tensor dataset partisi klien."""
    train_loader, _, train_ds, _ = build_client_dataloaders(partition_id=0, batch_size=2)
    assert len(train_ds) > 0

    batch = next(iter(train_loader))
    assert "audio" in batch
    assert "visual" in batch
    assert "text" in batch
    assert "phq8_score" in batch

    # Verifikasi shape tensor batch
    assert batch["audio"].shape == (2, 512, 80)
    assert batch["visual"].shape == (2, 512, 72, 3)
    assert batch["text"].shape == (2, 512, 768)
    assert batch["phq8_score"].shape == (2,)

    # Verifikasi tipe data
    assert batch["audio"].dtype == torch.float32
    assert batch["visual"].dtype == torch.float32
    assert batch["text"].dtype == torch.float32
    assert batch["phq8_score"].dtype == torch.float32


def test_global_test_dataloader():
    """Memverifikasi pemuatan DataLoader evaluasi terpusat server."""
    test_loader, test_ds = build_global_test_dataloader(batch_size=4)
    assert len(test_ds) == 56  # 56 sampel E-DAIC Test Set
    batch = next(iter(test_loader))
    assert batch["audio"].shape == (4, 512, 80)
    assert batch["visual"].shape == (4, 512, 72, 3)
    assert batch["text"].shape == (4, 512, 768)
    assert batch["phq8_score"].shape == (4,)
