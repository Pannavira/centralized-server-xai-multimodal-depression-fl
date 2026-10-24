"""
scripts/prepare_multiseed_partitions.py
=======================================
Generates, validates, and freezes deterministic partition sets for all 5 scenarios
across all 5 target seeds: [42, 43, 44, 45, 46].

Enforces 100% Zero Leakage against global hold-out test set and shared validation cohort.
"""

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from federated.data import DEFAULT_PARTITION_DIR
from federated.partition_generator import generate_controlled_label_skew_partitions
from federated.partitions.combined_skew import generate_controlled_combined_skew_partitions
from federated.partitions.quantity_skew import generate_controlled_quantity_skew_partitions


def get_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_controlled_iid_partitions(
    source_iid_dir: Path,
    output_dir: Path,
    seed: int,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load 163 participants from M5.1 pool
    dfs = [pd.read_csv(source_iid_dir / f"client_{c}_train.csv") for c in (1, 2, 3)]
    all_train = pd.concat(dfs, ignore_index=True)
    assert len(all_train) == 163

    pos_df = all_train[all_train["phq8_score"] >= 10.0].sort_values("participant_id").reset_index(drop=True)
    neg_df = all_train[all_train["phq8_score"] < 10.0].sort_values("participant_id").reset_index(drop=True)
    assert len(pos_df) == 50
    assert len(neg_df) == 113

    if seed == 42:
        # Replicate exact M5.1 allocation
        for c in (1, 2, 3):
            shutil.copy2(source_iid_dir / f"client_{c}_train.csv", output_dir / f"client_{c}_train.csv")
            shutil.copy2(source_iid_dir / f"client_{c}_val.csv", output_dir / f"client_{c}_val.csv")
    else:
        rng = np.random.RandomState(seed)
        pos_perm = rng.permutation(len(pos_df))
        neg_perm = rng.permutation(len(neg_df))

        pos_shuffled = pos_df.iloc[pos_perm].reset_index(drop=True)
        neg_shuffled = neg_df.iloc[neg_perm].reset_index(drop=True)

        # 54 / 55 / 54 split
        # Positives: 16, 17, 17
        # Negatives: 38, 38, 37
        c1_df = pd.concat([pos_shuffled.iloc[:16], neg_shuffled.iloc[:38]], ignore_index=True)
        c2_df = pd.concat([pos_shuffled.iloc[16:33], neg_shuffled.iloc[38:76]], ignore_index=True)
        c3_df = pd.concat([pos_shuffled.iloc[33:50], neg_shuffled.iloc[76:113]], ignore_index=True)

        assert len(c1_df) == 54 and (c1_df["phq8_score"] >= 10.0).sum() == 16
        assert len(c2_df) == 55 and (c2_df["phq8_score"] >= 10.0).sum() == 17
        assert len(c3_df) == 54 and (c3_df["phq8_score"] >= 10.0).sum() == 17

        c1_df.to_csv(output_dir / "client_1_train.csv", index=False)
        c2_df.to_csv(output_dir / "client_2_train.csv", index=False)
        c3_df.to_csv(output_dir / "client_3_train.csv", index=False)

        # Shared validation cohort
        for c in (1, 2, 3):
            shutil.copy2(source_iid_dir / f"client_{c}_val.csv", output_dir / f"client_{c}_val.csv")

    manifest = {
        "scenario": "IID_3Clients",
        "seed": seed,
        "protocol": "controlled_iid_multiseed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "client_train": {
            f"client_{c}": {
                "file": f"client_{c}_train.csv",
                "sha256": get_file_sha256(output_dir / f"client_{c}_train.csv"),
                "samples": len(pd.read_csv(output_dir / f"client_{c}_train.csv")),
                "positives": int((pd.read_csv(output_dir / f"client_{c}_train.csv")["phq8_score"] >= 10.0).sum()),
            }
            for c in (1, 2, 3)
        },
    }
    with open(output_dir / "partition_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def verify_partition_leakage(part_dir: Path, test_index_path: Path) -> None:
    test_df = pd.read_csv(test_index_path)
    test_ids = set(test_df["participant_id"].astype(str))

    train_ids: Set[str] = set()
    total_train_samples = 0
    for c in (1, 2, 3):
        t_path = part_dir / f"client_{c}_train.csv"
        df_c = pd.read_csv(t_path)
        pids = set(df_c["participant_id"].astype(str))
        # Ensure no overlap with other clients
        overlap = train_ids.intersection(pids)
        assert len(overlap) == 0, f"Cross-client overlap in {part_dir}: {overlap}"
        train_ids.update(pids)
        total_train_samples += len(df_c)

    assert total_train_samples == 163, f"Total train samples != 163 in {part_dir}: {total_train_samples}"
    assert len(train_ids) == 163, f"Unique train pids != 163 in {part_dir}: {len(train_ids)}"

    leakage = train_ids.intersection(test_ids)
    assert len(leakage) == 0, f"CRITICAL LEAKAGE in {part_dir}: {leakage}"


def main():
    print("=" * 70)
    print("PREPARING MULTI-SEED PARTITIONS (M5.6)")
    print("=" * 70)

    base_part_dir = Path("storage/federated/multiseed/partitions")
    base_part_dir.mkdir(parents=True, exist_ok=True)

    source_iid_dir = Path("C:/Wira/Kuliah/Penelitian/clinical-xai-multimodal-depression-fl/federated_learning_output/IID_3Clients")
    test_index_path = Path("C:/Wira/Kuliah/Penelitian/clinical-xai-multimodal-depression-fl/federated_learning_output/global_test_index.csv")

    seeds = [42, 43, 44, 45, 46]
    scenarios = [
        "IID_3Clients",
        "non_iid_label_skew",
        "non_iid_quantity_skew",
        "non_iid_modality_skew",
        "non_iid_combined",
    ]

    partition_registry = {}

    for s in seeds:
        print(f"\n>>> Generating Partitions for Seed {s} <<<")
        seed_part_base = base_part_dir / f"seed_{s}"
        seed_part_base.mkdir(parents=True, exist_ok=True)

        for scen in scenarios:
            scen_dir = seed_part_base / scen
            scen_dir.mkdir(parents=True, exist_ok=True)

            if scen == "IID_3Clients":
                manifest = generate_controlled_iid_partitions(source_iid_dir, scen_dir, seed=s)
            elif scen == "non_iid_modality_skew":
                # Modality skew uses the identical participant partitioning as IID for that seed
                iid_dir = seed_part_base / "IID_3Clients"
                for c in (1, 2, 3):
                    shutil.copy2(iid_dir / f"client_{c}_train.csv", scen_dir / f"client_{c}_train.csv")
                    shutil.copy2(iid_dir / f"client_{c}_val.csv", scen_dir / f"client_{c}_val.csv")
                shutil.copy2(iid_dir / "partition_manifest.json", scen_dir / "partition_manifest.json")
            elif scen == "non_iid_label_skew":
                if s == 42:
                    hist_ls = Path("storage/federated/partitions/label_skew_seed42")
                    for f in hist_ls.glob("*.csv"):
                        shutil.copy2(f, scen_dir / f.name)
                    for f in hist_ls.glob("*.json"):
                        shutil.copy2(f, scen_dir / f.name)
                else:
                    generate_controlled_label_skew_partitions(
                        source_iid_dir=source_iid_dir,
                        output_dir=scen_dir,
                        seed=s,
                        p1=43,
                        p2=2,
                        p3=5,
                    )
            elif scen == "non_iid_quantity_skew":
                if s == 42:
                    hist_qs = Path("storage/federated/partitions/quantity_skew_seed42")
                    for f in hist_qs.glob("*.csv"):
                        shutil.copy2(f, scen_dir / f.name)
                    for f in hist_qs.glob("*.json"):
                        shutil.copy2(f, scen_dir / f.name)
                else:
                    generate_controlled_quantity_skew_partitions(
                        source_iid_dir=source_iid_dir,
                        output_dir=scen_dir,
                        seed=s,
                        target_quantities=(98, 49, 16),
                    )
            elif scen == "non_iid_combined":
                if s == 42:
                    hist_cs = Path("storage/federated/partitions/combined_skew_seed42")
                    for f in hist_cs.glob("*.csv"):
                        shutil.copy2(f, scen_dir / f.name)
                    for f in hist_cs.glob("*.json"):
                        shutil.copy2(f, scen_dir / f.name)
                else:
                    generate_controlled_combined_skew_partitions(
                        source_iid_dir=source_iid_dir,
                        output_dir=scen_dir,
                        seed=s,
                        target_quantities=(98, 49, 16),
                        target_positives=(18, 19, 13),
                    )

            # Ensure validation cohort files exist
            for c in (1, 2, 3):
                v_dst = scen_dir / f"client_{c}_val.csv"
                if not v_dst.exists():
                    shutil.copy2(source_iid_dir / f"client_{c}_val.csv", v_dst)

            # Leakage & cardinality validation
            verify_partition_leakage(scen_dir, test_index_path)

            key = f"seed_{s}_{scen}"
            manifest_file = scen_dir / "partition_manifest.json"
            manifest_sha = get_file_sha256(manifest_file) if manifest_file.exists() else "NO_MANIFEST"

            c1_sha = get_file_sha256(scen_dir / "client_1_train.csv")
            c2_sha = get_file_sha256(scen_dir / "client_2_train.csv")
            c3_sha = get_file_sha256(scen_dir / "client_3_train.csv")

            partition_registry[key] = {
                "seed": s,
                "scenario": scen,
                "partition_dir": str(scen_dir).replace("\\", "/"),
                "manifest_sha256": manifest_sha,
                "client_1_sha256": c1_sha,
                "client_2_sha256": c2_sha,
                "client_3_sha256": c3_sha,
                "zero_leakage_verified": True,
            }
            print(f"  [{scen}] Verified: 163 participants, Zero Leakage, SHA: {manifest_sha[:16]}...")

    # Save registry
    reg_path = base_part_dir / "partitions_registry.json"
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(partition_registry, f, indent=2)

    print(f"\n[DONE] All 25 partition sets generated and verified: {reg_path}")


if __name__ == "__main__":
    main()
