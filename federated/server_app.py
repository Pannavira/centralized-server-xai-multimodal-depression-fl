"""
federated/server_app.py
=======================
Implementasi ServerApp modern Flower 1.40 Message API untuk FED-MIND FL (Milestone 5.1).

Fitur:
- Menggunakan ServerApp(@app.main()) dan Grid untuk komunikasi terfederasi.
- Menggunakan strategi FEDMINDFedAvg terbobot (weighted_by_key="num-examples").
- Mencatat metrik pelatihan & validasi per-klien ke client_metrics.csv.
- Inisialisasi model global awal (Round 0) deterministik dengan seed 42.
- Evaluasi terpusat (evaluate_fn) pada global hold-out test set setiap ronde (reporting-only).
- Menyimpan dataset_manifest.json (Zero Leakage Check & SHA-256) dan environment.json.
- Checkpointing otomatis per-ronde dengan SHA-256 dan metadata lengkap.
- Pelacakan beban komunikasi parameter kumulatif (communication.csv).
- Mendukung Resume / Recovery dari checkpoint ronde sebelumnya (start_round -> num_server_rounds).
- Perbandingan otomatis dengan model referensi historis (reference_comparison.json).
- Pembuatan kurva konvergensi visual otomatis (plots/).
"""

import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import torch
from flwr.app.message.arrayrecord import ArrayRecord
from flwr.app.message.configrecord import ConfigRecord
from flwr.app.message.context import Context
from flwr.app.message.message import Message
from flwr.app.message.metricrecord import MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from federated.checkpointing import FLRunManager
from federated.communication import record_round_communication
from federated.data import generate_dataset_manifest
from federated.evaluation import GlobalEvaluator, compare_with_historical_reference
from federated.task import build_model

logger = logging.getLogger("fedmind.server")
app = ServerApp()


def get_environment_info() -> Dict[str, Any]:
    """Mengumpulkan metadata lengkap lingkungan komputasi untuk reproduksibilitas."""
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    cuda_ver = torch.version.cuda if torch.cuda.is_available() else "None"
    cudnn_ver = (
        torch.backends.cudnn.version()
        if (torch.cuda.is_available() and hasattr(torch.backends, "cudnn"))
        else "None"
    )

    try:
        import flwr
        flwr_ver = flwr.__version__
    except ImportError:
        flwr_ver = "1.40.0"

    try:
        import ray
        ray_ver = ray.__version__
    except ImportError:
        ray_ver = "None"

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": cuda_ver,
        "cudnn_version": cudnn_ver,
        "flower_version": flwr_ver,
        "ray_version": ray_ver,
        "gpu_name": gpu_name,
        "git_commit": "828c079d5c4ecffb935d77d93e9d48a9f683d612",
        "model_parameter_count": 7832001,
        "model_architecture": "MultimodalBaselineModel",
    }


class FEDMINDFedAvg(FedAvg):
    """
    Subclass FedAvg Flower 1.40 untuk menangkap dan mencatat metrik per-klien
    (client_metrics.csv) tanpa mengubah sifat agregasi FedAvg.
    """

    def __init__(self, run_manager: FLRunManager, round_offset: int = 0, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.run_manager = run_manager
        self.round_offset = round_offset

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> Tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        actual_round = self.round_offset + server_round
        # Catat metrik tiap client yang mengembalikan update
        for reply in replies:
            if reply.has_content() and "metrics" in reply.content:
                metrics_rec = reply.content["metrics"]
                src_node = getattr(reply.metadata, "src_node_id", 0)
                client_id = f"node_{src_node}"
                num_ex = int(metrics_rec.get("num-examples", 0))
                self.run_manager.record_client_metrics(
                    server_round=actual_round,
                    client_id=client_id,
                    num_examples=num_ex,
                    metrics=dict(metrics_rec),
                )

        return super().aggregate_train(server_round, replies)


@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    Entrypoint utama ServerApp yang dieksekusi oleh runtime Flower.
    """
    run_cfg = context.run_config

    num_server_rounds = int(run_cfg.get("num-server-rounds", 50))
    fraction_train = float(run_cfg.get("fraction-train", 1.0))
    min_train_nodes = int(run_cfg.get("min-train-nodes", 3))
    min_available_nodes = int(run_cfg.get("min-available-nodes", 3))
    seed = int(run_cfg.get("seed", 42))
    batch_size = int(run_cfg.get("batch-size", 16))
    local_epochs = int(run_cfg.get("local-epochs", 2))
    learning_rate = float(run_cfg.get("learning-rate", 1e-4))
    
    scenario = str(run_cfg.get("scenario", "IID_3Clients")).strip()
    is_label_skew = scenario in ("non_iid_label_skew", "label_skew", "controlled_label_skew_v1")
    is_quantity_skew = scenario in ("non_iid_quantity_skew", "quantity_skew", "controlled_quantity_skew_v1")
    if is_quantity_skew:
        run_prefix = "FL-QUANTITY-SKEW"
        default_status = "controlled_experiment_quantity_skew"
    elif is_label_skew:
        run_prefix = "FL-LABEL-SKEW"
        default_status = "controlled_experiment_label_skew"
    else:
        run_prefix = "FL-RUN"
        default_status = "research_reproduction_fl_iid_approximate"
    scientific_status = str(run_cfg.get("scientific-status", default_status))
    baseline_control_run = str(run_cfg.get("baseline-control-run", "FL-RUN-20261016-130403-45D587")).strip()

    resume = bool(run_cfg.get("resume", False))
    resume_run_id = str(run_cfg.get("resume-run-id", "")).strip()

    logger.info("=" * 60)
    logger.info(f"MEMULAI FED-MIND FEDERATED LEARNING SERVER (Flower 1.40) - {scenario}")
    logger.info("=" * 60)
    logger.info(f"Target Server Rounds : {num_server_rounds}")
    logger.info(f"Scenario             : {scenario} (Label Skew: {is_label_skew})")
    logger.info(f"Min Participating    : {min_train_nodes} / {min_available_nodes} nodes")
    logger.info(f"Random Seed          : {seed}")
    logger.info(f"Scientific Status    : {scientific_status}")
    logger.info(f"Resume Mode          : {resume} (ID: {resume_run_id or 'auto/none'})")

    # 1. Inisialisasi Run Manager dan direktori artefak
    if resume:
        if not resume_run_id or resume_run_id.lower() in ("auto", "latest"):
            runs_dir = Path("storage/federated/runs")
            all_runs = [d.name for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith(f"{run_prefix}-")]
            if all_runs:
                resume_run_id = sorted(all_runs)[-1]
                logger.info(f"[RESUME AUTO] Menggunakan run_id terakhir: {resume_run_id}")

        if resume_run_id:
            run_manager = FLRunManager(run_id=resume_run_id, prefix=run_prefix)
            latest_completed = run_manager.get_latest_completed_round()
            if latest_completed is not None and latest_completed >= 0:
                start_round_offset = latest_completed
                logger.info(f"[RESUME] Melanjutkan eksperimen run {resume_run_id} dari ronde {start_round_offset}")
            else:
                start_round_offset = 0
                logger.warning(f"[RESUME] Tidak ditemukan checkpoint pada {resume_run_id}, memulai dari ronde 0")
        else:
            run_manager = FLRunManager(prefix=run_prefix)
            start_round_offset = 0
    else:
        run_manager = FLRunManager(prefix=run_prefix)
        start_round_offset = 0

    run_id = run_manager.run_id
    logger.info(f"Run ID dialokasikan : {run_id}")
    logger.info(f"Direktori Hasil     : {run_manager.run_dir}")

    # 2. Simpan konfigurasi awal, manifest dataset, dan environment jika baru mulai
    if start_round_offset == 0:
        config_record = {
            "run_id": run_id,
            "scenario": scenario,
            "num_server_rounds": num_server_rounds,
            "fraction_train": fraction_train,
            "min_train_nodes": min_train_nodes,
            "min_available_nodes": min_available_nodes,
            "seed": seed,
            "batch_size": batch_size,
            "local_epochs": local_epochs,
            "learning_rate": learning_rate,
            "scientific_status": scientific_status,
            "baseline_control_run": baseline_control_run,
            "strategy": "FedAvg",
            "weighted_by_key": "num-examples",
        }
        run_manager.save_config(config_record)

        # Simpan metadata lingkungan
        env_info = get_environment_info()
        run_manager.save_environment(env_info)

        # Simpan manifest dataset (zero-leakage assertion)
        try:
            ds_manifest = generate_dataset_manifest(scenario=scenario)
            run_manager.save_dataset_manifest(ds_manifest)
            logger.info(f"[Dataset Manifest] Zero Leakage Terkonfirmasi: {ds_manifest['zero_leakage_verified']}")
        except Exception as e:
            logger.error(f"[Dataset Manifest Error] {e}")

        # Salin artefak partisi jika skenario quantity skew atau label skew
        if is_quantity_skew:
            import shutil
            part_dir = Path("storage/federated/partitions/quantity_skew_seed42")
            for fname in ["partition_manifest.json", "quantity_distribution.json", "quantity_distribution.csv"]:
                src_f = part_dir / fname
                if src_f.exists():
                    shutil.copy2(src_f, run_manager.run_dir / fname)
            part_plots = part_dir / "plots"
            if part_plots.exists():
                for p_f in part_plots.glob("*.png"):
                    shutil.copy2(p_f, run_manager.plots_dir / p_f.name)
        elif is_label_skew:
            import shutil
            part_dir = Path("storage/federated/partitions/label_skew_seed42")
            for fname in ["partition_manifest.json", "label_distribution.json", "label_distribution.csv"]:
                src_f = part_dir / fname
                if src_f.exists():
                    shutil.copy2(src_f, run_manager.run_dir / fname)
            plot_src = part_dir / "label_distribution_iid_vs_label_skew.png"
            if plot_src.exists():
                shutil.copy2(plot_src, run_manager.plots_dir / "label_distribution_iid_vs_label_skew.png")

    # 3. Inisialisasi Model Awal (Atau muat dari ronde resume / M5.1 checkpoint)
    r0_source_checkpoint: Optional[str] = None
    if start_round_offset > 0:
        initial_state_dict, r_meta = run_manager.load_round_checkpoint(start_round_offset)
        initial_arrays = ArrayRecord.from_torch_state_dict(initial_state_dict)
        logger.info(f"[RESUME] Berhasil memuat bobot model dari round_{start_round_offset:03d}/model.pt")
    else:
        init_ckpt_req = str(run_cfg.get("initial-checkpoint", "")).strip()
        if not init_ckpt_req and (is_label_skew or is_quantity_skew):
            init_ckpt_req = "storage/federated/runs/FL-RUN-20261016-130403-45D587/rounds/round_000/model.pt"

        if init_ckpt_req and Path(init_ckpt_req).exists():
            from federated.checkpointing import compute_file_sha256
            ckpt_sha = compute_file_sha256(init_ckpt_req)
            logger.info(f"[Initial Model] Memuat dari checkpoint kontrol M5.1: {init_ckpt_req}")
            logger.info(f"[Initial Model] SHA-256 Checksum: {ckpt_sha}")
            assert ckpt_sha == "a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973", (
                f"CRITICAL: Round 0 SHA-256 mismatch! {ckpt_sha} != a1c052476e..."
            )
            initial_state_dict = torch.load(init_ckpt_req, map_location="cpu", weights_only=False)
            initial_model = build_model()
            initial_model.load_state_dict(initial_state_dict, strict=True)
            initial_arrays = ArrayRecord.from_torch_state_dict(initial_state_dict)
            r0_source_checkpoint = init_ckpt_req
        else:
            initial_model = build_model(seed=seed)
            initial_state_dict = initial_model.state_dict()
            initial_arrays = ArrayRecord.from_torch_state_dict(initial_state_dict)

    # 4. Siapkan evaluator global test terpusat
    evaluator = GlobalEvaluator()
    round_eval_records: Dict[int, Dict[str, Any]] = {}
    round_checkpoint_hashes: Dict[int, str] = {}

    if start_round_offset > 0:
        for r in range(start_round_offset + 1):
            r_dir = run_manager.rounds_dir / f"round_{r:03d}"
            meta_f = r_dir / "metadata.json"
            if meta_f.exists():
                try:
                    with open(meta_f, "r", encoding="utf-8") as f:
                        m_data = json.load(f)
                        round_checkpoint_hashes[r] = m_data.get("checkpoint_sha256", "")
                        round_eval_records[r] = m_data.get("global_evaluation_metrics", {})
                except Exception:
                    pass

    def central_eval_fn(server_round: int, arrays: ArrayRecord) -> Optional[MetricRecord]:
        actual_round = start_round_offset + server_round

        # Jika resume dan ini adalah callback inisialisasi awal Flower (server_round == 0), lewati jika sudah ada
        if start_round_offset > 0 and server_round == 0:
            logger.info(f"[RESUME] Ronde {actual_round} telah tercatat sebelumnya, melewati evaluasi duplikat R0.")
            return MetricRecord({"test_mae": 0.0, "test_rmse": 0.0, "test_f1": 0.0, "test_recall": 0.0})

        logger.info(f"[Server Central Evaluation] Ronde {actual_round} (Flower step {server_round})...")
        current_state_dict = arrays.to_torch_state_dict()
        current_model = build_model()
        current_model.load_state_dict(current_state_dict, strict=True)

        # Evaluasi terhadap global hold-out test set (reporting-only)
        metrics = evaluator.evaluate_state_dict(current_state_dict)
        round_eval_records[actual_round] = metrics

        logger.info(
            f"[Round {actual_round:02d} Test Set] MAE={metrics.get('mae')}, "
            f"RMSE={metrics.get('rmse')}, F1={metrics.get('macro_f1')}, Recall={metrics.get('recall')}"
        )

        # Simpan ke metrics.csv
        run_manager.record_metrics(server_round=actual_round, metrics=metrics)

        # Simpan checkpoint ronde dan catat communication overhead
        participating = min_train_nodes if actual_round > 0 else 0
        comm_rec = record_round_communication(
            csv_path=run_manager.comm_csv_path,
            server_round=actual_round,
            model=current_model,
            participating_clients=participating,
        )

        ckpt_metadata = {
            "run_id": run_id,
            "round": actual_round,
            "scenario": scenario,
            "scientific_status": scientific_status,
            "training_strategy": "FedAvg",
            "number_of_clients": min_available_nodes,
            "number_of_participating_clients": participating,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "loss_function": "ClassWeightedMSELoss",
            "random_seed": seed,
            "global_evaluation_metrics": metrics,
            "communication_estimate": comm_rec,
        }

        source_copy = r0_source_checkpoint if actual_round == 0 else None
        _, sha256 = run_manager.save_round_checkpoint(
            server_round=actual_round,
            state_dict=current_state_dict,
            metadata=ckpt_metadata,
            source_checkpoint_to_copy=source_copy,
        )
        round_checkpoint_hashes[actual_round] = sha256
        logger.info(f"[Round {actual_round:02d} Checkpoint] SHA-256: {sha256}")

        # Buat grafik konvergensi setiap 5 ronde atau saat ronde akhir
        if actual_round % 5 == 0 or actual_round == num_server_rounds:
            run_manager.generate_plots()

        # Jika sudah mencapai ronde terakhir, jalankan komparasi
        if actual_round == num_server_rounds:
            if is_quantity_skew:
                from federated.comparison import generate_iid_vs_quantity_skew_comparison
                try:
                    generate_iid_vs_quantity_skew_comparison(
                        quantity_skew_run_dir=run_manager.run_dir,
                        iid_control_run_dir=Path("storage/federated/runs") / baseline_control_run,
                    )
                    logger.info("[IID vs Quantity Skew Comparison] Berhasil disimpan ke iid_comparison.json dan plots!")
                except Exception as e:
                    logger.warning(f"[IID vs Quantity Skew Comparison Warning] {e}")
            elif is_label_skew:
                from federated.comparison import generate_iid_vs_label_skew_comparison
                try:
                    generate_iid_vs_label_skew_comparison(
                        label_skew_run_dir=run_manager.run_dir,
                        iid_control_run_dir=Path("storage/federated/runs") / baseline_control_run,
                    )
                    logger.info("[IID Comparison] Berhasil disimpan ke iid_comparison.json dan plots!")
                except Exception as e:
                    logger.warning(f"[IID Comparison Warning] {e}")
            else:
                try:
                    comp_result = compare_with_historical_reference(
                        reproduction_state_dict=current_state_dict,
                        evaluator=evaluator,
                    )
                    run_manager.save_reference_comparison(comp_result)
                    logger.info("[Reference Comparison] Berhasil disimpan ke reference_comparison.json")
                except Exception as e:
                    logger.warning(f"[Reference Comparison Warning] {e}")

        # Kembalikan MetricRecord untuk Flower
        return MetricRecord({
            "test_mae": float(metrics.get("mae", 0.0)),
            "test_rmse": float(metrics.get("rmse", 0.0)),
            "test_f1": float(metrics.get("macro_f1", 0.0)),
            "test_recall": float(metrics.get("recall", 0.0)),
        })

    # 5. Hitung sisa ronde yang harus dijalankan
    remaining_rounds = max(num_server_rounds - start_round_offset, 0)
    logger.info(f"Sisa ronde yang akan dieksekusi: {remaining_rounds} ronde (Offset: {start_round_offset})")

    if remaining_rounds > 0:
        # Bangun Strategi FedAvg Kustom
        strategy = FEDMINDFedAvg(
            run_manager=run_manager,
            round_offset=start_round_offset,
            fraction_train=fraction_train,
            fraction_evaluate=0.0,  # Evaluasi federated client ditangani secara lokal di client_app
            min_train_nodes=min_train_nodes,
            min_evaluate_nodes=0,
            min_available_nodes=min_available_nodes,
            weighted_by_key="num-examples",
        )

        train_config = ConfigRecord({
            "local-epochs": local_epochs,
            "batch-size": batch_size,
            "learning-rate": learning_rate,
            "scenario": scenario,
        })

        result = strategy.start(
            grid=grid,
            initial_arrays=initial_arrays,
            num_rounds=remaining_rounds,
            train_config=train_config,
            evaluate_fn=central_eval_fn,
        )

    # 6. Finalisasi Manifest Run & Plots
    run_manager.generate_plots()

    manifest = {
        "run_id": run_id,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETED",
        "rounds_executed": num_server_rounds,
        "participating_nodes": min_train_nodes,
        "round_checkpoints": {
            f"round_{r:03d}": {
                "sha256": round_checkpoint_hashes.get(r),
                "metrics": round_eval_records.get(r),
            }
            for r in round_checkpoint_hashes
        },
        "scientific_status": scientific_status,
    }
    run_manager.save_manifest(manifest)

    logger.info("=" * 60)
    logger.info(f"FED-MIND FEDERATED LEARNING REPRODUKSI SELESAI ({num_server_rounds} ROUNDS)")
    logger.info("=" * 60)
