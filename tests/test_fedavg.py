"""
tests/test_fedavg.py
====================
Pengujian Matematis Eksak Agregasi FedAvg (Milestone 4).

Sesuai instruksi:
- Menggunakan tensor parameter deterministik artifisial kecil:
  Klien 1: n = 10
  Klien 2: n = 20
  Klien 3: n = 30
- Memverifikasi bahwa tensor agregasi identik dengan rata-rata terbobot manual.
- Menguji baik kalkulasi matematis murni maupun implementasi Flower aggregate_arrayrecords.
"""

import pytest
import torch
from flwr.app.message.arrayrecord import ArrayRecord
from flwr.app.message.metricrecord import MetricRecord
from flwr.app.message.recorddict import RecordDict
from flwr.serverapp.strategy.fedavg import aggregate_arrayrecords


def manual_fedavg(
    client_params: list[torch.Tensor],
    client_samples: list[int],
) -> torch.Tensor:
    """Kalkulasi manual matematis murni untuk FedAvg."""
    total_samples = sum(client_samples)
    weighted_sum = torch.zeros_like(client_params[0], dtype=torch.float64)
    for p, n in zip(client_params, client_samples):
        weighted_sum += p.double() * (n / total_samples)
    return weighted_sum.float()


def test_fedavg_mathematical_correctness():
    """
    Verifikasi matematis FedAvg dengan 3 klien: n = [10, 20, 30].
    Klien 1: w1 = [1.0, 2.0, 3.0], n1 = 10 (bobot 10/60 = 1/6)
    Klien 2: w2 = [4.0, 5.0, 6.0], n2 = 20 (bobot 20/60 = 2/6)
    Klien 3: w3 = [7.0, 8.0, 9.0], n3 = 30 (bobot 30/60 = 3/6)

    Perhitungan Manual Eksak:
    idx 0: 1*(1/6) + 4*(2/6) + 7*(3/6) = (1 + 8 + 21)/6 = 30/6 = 5.0
    idx 1: 2*(1/6) + 5*(2/6) + 8*(3/6) = (2 + 10 + 24)/6 = 36/6 = 6.0
    idx 2: 3*(1/6) + 6*(2/6) + 9*(3/6) = (3 + 12 + 27)/6 = 42/6 = 7.0
    Hasil Eksak = [5.0, 6.0, 7.0]
    """
    t1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    t2 = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)
    t3 = torch.tensor([7.0, 8.0, 9.0], dtype=torch.float32)

    samples = [10, 20, 30]
    expected_manual = torch.tensor([5.0, 6.0, 7.0], dtype=torch.float32)

    # 1. Verifikasi kalkulasi formula manual independen
    manual_res = manual_fedavg([t1, t2, t3], samples)
    assert torch.allclose(manual_res, expected_manual, atol=1e-6)

    # 2. Verifikasi Flower 1.40 aggregate_arrayrecords
    rd1 = RecordDict({
        "arrays": ArrayRecord.from_torch_state_dict({"weight": t1}),
        "metrics": MetricRecord({"num-examples": samples[0]}),
    })
    rd2 = RecordDict({
        "arrays": ArrayRecord.from_torch_state_dict({"weight": t2}),
        "metrics": MetricRecord({"num-examples": samples[1]}),
    })
    rd3 = RecordDict({
        "arrays": ArrayRecord.from_torch_state_dict({"weight": t3}),
        "metrics": MetricRecord({"num-examples": samples[2]}),
    })

    flower_aggr = aggregate_arrayrecords(
        records=[rd1, rd2, rd3],
        weighting_metric_name="num-examples",
    )
    flower_state = flower_aggr.to_torch_state_dict()
    flower_weight = flower_state["weight"]

    # Pastikan hasil agregasi Flower identik dengan kalkulasi manual eksak
    assert torch.allclose(flower_weight, expected_manual, atol=1e-6), (
        f"Agregasi Flower ({flower_weight}) tidak sama dengan ekspektasi manual ({expected_manual})"
    )


def test_fedavg_multi_dimensional_tensor():
    """Verifikasi agregasi pada tensor multidimensi (matriks 2D)."""
    m1 = torch.ones((3, 4), dtype=torch.float32) * 2.0   # n=25
    m2 = torch.ones((3, 4), dtype=torch.float32) * 6.0   # n=75
    # Total = 100, bobot m1 = 0.25, m2 = 0.75
    # Hasil = 2.0*0.25 + 6.0*0.75 = 0.5 + 4.5 = 5.0

    rd1 = RecordDict({
        "arrays": ArrayRecord.from_torch_state_dict({"fc.weight": m1}),
        "metrics": MetricRecord({"num-examples": 25}),
    })
    rd2 = RecordDict({
        "arrays": ArrayRecord.from_torch_state_dict({"fc.weight": m2}),
        "metrics": MetricRecord({"num-examples": 75}),
    })

    flower_aggr = aggregate_arrayrecords(
        records=[rd1, rd2],
        weighting_metric_name="num-examples",
    )
    res = flower_aggr.to_torch_state_dict()["fc.weight"]
    expected = torch.ones((3, 4), dtype=torch.float32) * 5.0
    assert torch.allclose(res, expected, atol=1e-6)
