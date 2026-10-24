"""
federated/modality_degradation.py
=================================
Modul Transformasi dan Degradasi Modalitas Sintetis untuk Eksperimen Non-IID Modality Skew (Milestone 5.4).

Protokol Ilmiah: `controlled_modality_skew_v1`
- Client 1 (Partition 0, 54 sampel): Full Clean (Audio, Visual, Text normal tanpa degradasi).
- Client 2 (Partition 1, 55 sampel): Noisy Audio (Gaussian perturbation sigma=0.25 pada feature space ternormalisasi).
- Client 3 (Partition 2, 54 sampel): Degraded Visual (Temporal frame dropout 20% dengan mean-imputation referensi netral).
- Teks (Text): Selalu utuh dan bersih di ketiga klien (invarian kontrol).
- Shared Validation & Global Test: Selalu bersih 100% tanpa modifikasi.
- Determinisme: Menggunakan generator lokal ber-seed SHA-256 independen per subjek.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger("fedmind.modality_degradation")


@dataclass
class ModalityDegradationConfig:
    """Konfigurasi parameter eksperimen Modality Skew."""
    scenario: str = "non_iid_modality_skew"
    protocol_version: str = "controlled_modality_skew_v1"
    audio_noise_sigma: float = 0.25
    visual_frame_dropout_rate: float = 0.20
    seed: int = 42
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModalityDegradationConfig":
        valid_keys = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid_keys)


def validate_degradation_config(config: ModalityDegradationConfig) -> bool:
    """Memvalidasi integritas nilai parameter konfigurasi degradasi."""
    if config.audio_noise_sigma < 0.0:
        raise ValueError(f"audio_noise_sigma harus >= 0, didapat: {config.audio_noise_sigma}")
    if not (0.0 <= config.visual_frame_dropout_rate < 1.0):
        raise ValueError(
            f"visual_frame_dropout_rate harus dalam rentang [0.0, 1.0), didapat: {config.visual_frame_dropout_rate}"
        )
    return True


def get_deterministic_sample_seed(
    base_seed: int,
    client_id: Union[int, str],
    sample_id: Union[int, str],
    modality: str,
    protocol_version: str = "controlled_modality_skew_v1",
) -> int:
    """
    Menghasilkan seed 32-bit deterministik berbasis SHA-256 untuk subjek tertentu.
    Tidak menggunakan hash() bawaan Python untuk menghindari process-randomization salt.
    """
    key = f"{base_seed}_{client_id}_{sample_id}_{modality}_{protocol_version}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    # Ambil 8 karakter pertama hexdigest -> uint32 (0 s.d 4,294,967,295)
    return int(h[:8], 16)


def get_visual_neutral_frame(
    device: torch.device = torch.device("cpu"),
    fallback_path: Optional[Union[str, Path]] = None,
) -> torch.Tensor:
    """
    Memuat representasi frame visual netral berbasis mean training seluruh populasi subjek (72 nodes, 3 coords).
    """
    candidates = [
        Path(fallback_path) if fallback_path else None,
        Path("storage/models/visual_mean_frame.pt"),
        Path(__file__).resolve().parent.parent / "storage" / "models" / "visual_mean_frame.pt",
    ]
    for cand in candidates:
        if cand and cand.exists():
            try:
                frame = torch.load(cand, map_location=device, weights_only=True)
                if isinstance(frame, torch.Tensor) and frame.shape == (72, 3):
                    return frame.float()
            except Exception:
                pass

    # Fallback deterministik berbasis statistik populasi training E-DAIC 163 subjek (mean scalar = 0.2278)
    return torch.full((72, 3), fill_value=0.227831, dtype=torch.float32, device=device)


def apply_audio_noise(
    audio_tensor: torch.Tensor,
    sigma: float = 0.25,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Menambahkan perturbation Gaussian (mean=0, std=sigma) pada feature space audio ternormalisasi.
    HANYA diaplikasikan pada frame/timestep valid (bukan padding zero yang eksplisit).
    Tidak mengubah dimensi tensor atau tipe data.
    """
    if sigma <= 0.0:
        return audio_tensor.clone(), {
            "applied": False,
            "mean_perturbation": 0.0,
            "std_perturbation": 0.0,
            "valid_timesteps": audio_tensor.shape[0],
        }

    out = audio_tensor.clone().float()
    T, D = out.shape

    # Deteksi valid timesteps: baris yang tidak seluruhnya bernilai 0.0
    # (Pada representasi E-DAIC 512x80 bertile, seluruh 512 timestep aktif)
    row_norms = torch.norm(out, p=2, dim=-1)
    valid_mask = row_norms > 1e-6
    valid_count = int(valid_mask.sum().item())

    # Jika karena suatu hal seluruh baris bernilai 0, anggap seluruh timestep valid
    if valid_count == 0:
        valid_mask = torch.ones(T, dtype=torch.bool)
        valid_count = T

    # Generator lokal terisolasi: sama sekali TIDAK menyentuh global RNG
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    noise = torch.randn(out.shape, generator=gen, dtype=torch.float32) * sigma
    # Terapkan noise hanya pada timestep valid
    noise[~valid_mask] = 0.0

    out = out + noise

    actual_diff = out - audio_tensor.float()
    mean_abs_pert = float(torch.abs(actual_diff[valid_mask]).mean().item())
    observed_std = float(actual_diff[valid_mask].std().item())

    diagnostics = {
        "applied": True,
        "valid_timesteps": valid_count,
        "total_timesteps": T,
        "feature_dim": D,
        "target_sigma": sigma,
        "mean_absolute_perturbation": round(mean_abs_pert, 6),
        "observed_perturbation_std": round(observed_std, 6),
    }

    return out, diagnostics


def apply_visual_frame_dropout(
    visual_tensor: torch.Tensor,
    dropout_rate: float = 0.20,
    neutral_frame: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Menerapkan temporal frame dropout pada representasi visual:
    20% frame valid digantikan oleh frame referensi netral (mean-imputation).
    Dimensi tensor [512, 72, 3] dan tipe data dipertahankan secara utuh.
    """
    if dropout_rate <= 0.0:
        return visual_tensor.clone(), {
            "applied": False,
            "dropped_frames": 0,
            "valid_frames": visual_tensor.shape[0],
        }

    out = visual_tensor.clone().float()
    T = out.shape[0]

    # Identifikasi frame valid (frame yang tidak seluruhnya nol)
    frame_norms = torch.norm(out.reshape(T, -1), p=2, dim=-1)
    valid_indices = torch.where(frame_norms > 1e-6)[0]
    if len(valid_indices) == 0:
        valid_indices = torch.arange(T)

    num_valid = len(valid_indices)
    num_to_drop = max(1, int(round(num_valid * dropout_rate)))

    # Generator lokal terisolasi
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    # Permutasi acak indeks valid menggunakan generator lokal
    perm = torch.randperm(num_valid, generator=gen)
    drop_indices = valid_indices[perm[:num_to_drop]]

    if neutral_frame is None:
        neutral_frame = get_visual_neutral_frame(device=visual_tensor.device)
    neutral_frame = neutral_frame.to(visual_tensor.device).float()

    # Lakukan mean-imputation pada frame yang terpilih untuk di-drop
    out[drop_indices] = neutral_frame

    actual_diff = out - visual_tensor.float()
    mean_abs_diff = float(torch.abs(actual_diff).mean().item())

    diagnostics = {
        "applied": True,
        "total_frames": T,
        "valid_frames": num_valid,
        "dropped_frames": num_to_drop,
        "dropout_rate_target": dropout_rate,
        "dropout_rate_actual": round(num_to_drop / num_valid, 4),
        "mean_absolute_difference": round(mean_abs_diff, 6),
    }

    return out, diagnostics


def apply_client_modality_treatment(
    item: Dict[str, Any],
    partition_id: int,
    config: Optional[ModalityDegradationConfig] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Menerapkan perlakuan degradasi modalitas spesifik untuk partisi klien tertentu:
    - Partition 0 (Client 1): Clean (Audio, Visual, Text tidak berubah).
    - Partition 1 (Client 2): Audio Perturbation (Gaussian noise sigma=0.25).
    - Partition 2 (Client 3): Visual Frame Dropout (20% frame dropped dengan mean frame).
    - Text: Selalu bersih dan tidak dimodifikasi di seluruh klien.
    - PHQ-8 & Depression Label: Selalu bersih dan tidak dimodifikasi.
    """
    if config is None:
        config = ModalityDegradationConfig()

    out_item = dict(item)
    diag: Dict[str, Any] = {
        "partition_id": partition_id,
        "client_name": f"client_{partition_id + 1}",
        "audio_degraded": False,
        "visual_degraded": False,
        "text_degraded": False,
    }

    if not config.enabled:
        return out_item, diag

    pid = str(item.get("participant_id", "unknown"))

    # Client 1 (Partition 0): Full Clean
    if partition_id == 0:
        diag["treatment"] = "clean_full_modalities"
        return out_item, diag

    # Client 2 (Partition 1): Audio Perturbation
    elif partition_id == 1:
        seed = get_deterministic_sample_seed(
            base_seed=config.seed,
            client_id="client_2",
            sample_id=pid,
            modality="audio",
            protocol_version=config.protocol_version,
        )
        corrupted_audio, aud_diag = apply_audio_noise(
            audio_tensor=item["audio"],
            sigma=config.audio_noise_sigma,
            seed=seed,
        )
        out_item["audio"] = corrupted_audio
        diag["audio_degraded"] = True
        diag["audio_diagnostics"] = aud_diag
        diag["treatment"] = "audio_noisy"
        return out_item, diag

    # Client 3 (Partition 2): Visual Degradation
    elif partition_id == 2:
        seed = get_deterministic_sample_seed(
            base_seed=config.seed,
            client_id="client_3",
            sample_id=pid,
            modality="visual",
            protocol_version=config.protocol_version,
        )
        corrupted_visual, vis_diag = apply_visual_frame_dropout(
            visual_tensor=item["visual"],
            dropout_rate=config.visual_frame_dropout_rate,
            seed=seed,
        )
        out_item["visual"] = corrupted_visual
        diag["visual_degraded"] = True
        diag["visual_diagnostics"] = vis_diag
        diag["treatment"] = "visual_degraded"
        return out_item, diag

    else:
        # Default fallback jika ada partisi lain
        diag["treatment"] = "clean_unassigned"
        return out_item, diag


def compute_degradation_diagnostics(
    client_datasets: Dict[int, Any],
    config: Optional[ModalityDegradationConfig] = None,
) -> Dict[str, Any]:
    """
    Menjalankan audit diagnostik empiris komprehensif pada tensor data pelatihan klien
    untuk membuktikan degradasi modalitas diterapkan secara tepat dan terukur.
    """
    if config is None:
        config = ModalityDegradationConfig()

    summary: Dict[str, Any] = {
        "protocol_version": config.protocol_version,
        "seed": config.seed,
        "audio_noise_sigma": config.audio_noise_sigma,
        "visual_frame_dropout_rate": config.visual_frame_dropout_rate,
        "clients": {},
    }

    csv_rows = []

    for p_id in sorted(client_datasets.keys()):
        ds = client_datasets[p_id]
        c_name = f"client_{p_id + 1}"
        num_samples = len(ds)

        audio_diffs = []
        visual_diffs = []
        text_diffs = []
        dropped_frames_list = []

        for i in range(num_samples):
            # Ambil item mentah
            raw_item = ds._get_raw_item(i) if hasattr(ds, "_get_raw_item") else ds[i]
            # Terapkan treatment
            degraded_item, item_diag = apply_client_modality_treatment(
                item=raw_item,
                partition_id=p_id,
                config=config,
            )

            a_diff = float(torch.abs(degraded_item["audio"] - raw_item["audio"]).mean().item())
            v_diff = float(torch.abs(degraded_item["visual"] - raw_item["visual"]).mean().item())
            t_diff = float(torch.abs(degraded_item["text"] - raw_item["text"]).mean().item())

            audio_diffs.append(a_diff)
            visual_diffs.append(v_diff)
            text_diffs.append(t_diff)

            if "visual_diagnostics" in item_diag:
                dropped_frames_list.append(item_diag["visual_diagnostics"]["dropped_frames"])

            csv_rows.append({
                "client": c_name,
                "partition_id": p_id,
                "sample_idx": i,
                "participant_id": raw_item["participant_id"],
                "audio_mean_abs_diff": a_diff,
                "visual_mean_abs_diff": v_diff,
                "text_mean_abs_diff": t_diff,
            })

        mean_a_diff = float(np.mean(audio_diffs))
        mean_v_diff = float(np.mean(visual_diffs))
        mean_t_diff = float(np.mean(text_diffs))

        summary["clients"][c_name] = {
            "samples": num_samples,
            "audio_mean_abs_diff": round(mean_a_diff, 6),
            "visual_mean_abs_diff": round(mean_v_diff, 6),
            "text_mean_abs_diff": round(mean_t_diff, 6),
            "audio_modified": bool(mean_a_diff > 1e-6),
            "visual_modified": bool(mean_v_diff > 1e-6),
            "text_modified": bool(mean_t_diff > 1e-6),
        }

        if dropped_frames_list:
            summary["clients"][c_name]["mean_dropped_frames"] = round(float(np.mean(dropped_frames_list)), 2)
            summary["clients"][c_name]["dropped_frames_ratio"] = round(
                float(np.mean(dropped_frames_list)) / 512.0, 4
            )

    summary["diagnostics_dataframe"] = pd.DataFrame(csv_rows)
    return summary


def plot_modality_assignment(output_path: Union[str, Path]) -> None:
    """
    Membuat grafik visualisasi rancangan perlakuan modalitas per-klien (modality_assignment.png).
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    clients = ["Client 1\n(54 samples)", "Client 2\n(55 samples)", "Client 3\n(54 samples)"]
    modalities = ["Audio", "Visual", "Text"]

    # Status: 0 = Clean, 1 = Audio Noisy, 2 = Visual Degraded
    status_matrix = np.array([
        [0, 1, 0],  # Audio: C1 Clean, C2 Noisy, C3 Clean
        [0, 0, 2],  # Visual: C1 Clean, C2 Clean, C3 Degraded
        [0, 0, 0],  # Text: C1 Clean, C2 Clean, C3 Clean
    ])

    colors = {
        0: "#2ecc71",  # Clean (Green)
        1: "#e67e22",  # Audio Noisy (Orange)
        2: "#9b59b6",  # Visual Degraded (Purple)
    }

    labels = {
        0: "Clean (Original)",
        1: "Gaussian Noise (σ = 0.25)",
        2: "Temporal Frame Dropout (20%)",
    }

    for i in range(len(modalities)):
        for j in range(len(clients)):
            st = status_matrix[i, j]
            color = colors[st]
            rect = plt.Rectangle((j - 0.4, i - 0.35), 0.8, 0.7, color=color, alpha=0.9, zorder=2)
            ax.add_patch(rect)
            txt = "Clean" if st == 0 else ("Noisy Audio\n(σ = 0.25)" if st == 1 else "Degraded Visual\n(20% Dropout)")
            ax.text(
                j, i, txt,
                ha="center", va="center", color="white",
                fontweight="bold", fontsize=11, zorder=3,
            )

    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-0.6, 2.6)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(clients, fontsize=12, fontweight="bold")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(modalities, fontsize=12, fontweight="bold")
    ax.invert_yaxis()

    ax.set_title("FED-MIND M5.4: Controlled Non-IID Modality Skew Assignment", fontsize=14, fontweight="bold", pad=15)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Legend
    legend_patches = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in [0, 1, 2]]
    legend_labels = [labels[k] for k in [0, 1, 2]]
    ax.legend(
        legend_patches, legend_labels,
        loc="upper center", bbox_to_anchor=(0.5, -0.15),
        ncol=3, frameon=True, fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Modality assignment plot tersimpan di: {out_file}")


def plot_degradation_diagnostics(
    diagnostics: Dict[str, Any],
    output_path: Union[str, Path],
) -> None:
    """
    Membuat visualisasi bukti empiris degradasi modalitas (degradation_diagnostics.png).
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Panel A: Mean Absolute Feature Perturbation across Clients
    clients = ["Client 1 (Clean)", "Client 2 (Audio Noisy)", "Client 3 (Visual Degraded)"]
    audio_diffs = [
        diagnostics["clients"]["client_1"]["audio_mean_abs_diff"],
        diagnostics["clients"]["client_2"]["audio_mean_abs_diff"],
        diagnostics["clients"]["client_3"]["audio_mean_abs_diff"],
    ]
    visual_diffs = [
        diagnostics["clients"]["client_1"]["visual_mean_abs_diff"],
        diagnostics["clients"]["client_2"]["visual_mean_abs_diff"],
        diagnostics["clients"]["client_3"]["visual_mean_abs_diff"],
    ]
    text_diffs = [
        diagnostics["clients"]["client_1"]["text_mean_abs_diff"],
        diagnostics["clients"]["client_2"]["text_mean_abs_diff"],
        diagnostics["clients"]["client_3"]["text_mean_abs_diff"],
    ]

    x = np.arange(len(clients))
    width = 0.25

    ax0 = axes[0]
    ax0.bar(x - width, audio_diffs, width, label="Audio Mean Abs Diff", color="#e67e22")
    ax0.bar(x, visual_diffs, width, label="Visual Mean Abs Diff", color="#9b59b6")
    ax0.bar(x + width, text_diffs, width, label="Text Mean Abs Diff (Clean)", color="#2ecc71")

    ax0.set_xticks(x)
    ax0.set_xticklabels(clients, fontsize=10, fontweight="bold")
    ax0.set_ylabel("Mean Absolute Difference (Feature Space)", fontsize=11, fontweight="bold")
    ax0.set_title("A. Empirically Observed Modality Perturbation", fontsize=12, fontweight="bold")
    ax0.grid(axis="y", linestyle="--", alpha=0.5)
    ax0.legend(loc="upper left", fontsize=9)

    # Panel B: Target vs Observed Degradation Parameters
    ax1 = axes[1]
    metrics = ["Audio Noise Sigma", "Visual Frame Dropout Rate"]
    target_vals = [diagnostics["audio_noise_sigma"], diagnostics["visual_frame_dropout_rate"]]
    # Derived from C2 audio observed std and C3 dropout ratio
    c2_sigma = diagnostics["clients"]["client_2"].get("audio_mean_abs_diff", 0.0) / 0.79788  # E[|Z|] = sigma * sqrt(2/pi)
    c3_rate = diagnostics["clients"]["client_3"].get("dropped_frames_ratio", 0.20)
    observed_vals = [c2_sigma, c3_rate]

    x1 = np.arange(len(metrics))
    ax1.bar(x1 - 0.15, target_vals, 0.3, label="Target Protocol Setting", color="#34495e")
    ax1.bar(x1 + 0.15, observed_vals, 0.3, label="Empirically Realized", color="#3498db")

    ax1.set_xticks(x1)
    ax1.set_xticklabels(metrics, fontsize=11, fontweight="bold")
    ax1.set_ylabel("Parameter Value", fontsize=11, fontweight="bold")
    ax1.set_title("B. Target vs Realized Degradation Parameters", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Degradation diagnostics plot tersimpan di: {out_file}")
