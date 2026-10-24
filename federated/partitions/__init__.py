"""
federated/partitions/__init__.py
================================
Package modul partisi Non-IID untuk FED-MIND Federated Learning.
"""

from federated.partitions.quantity_skew import generate_controlled_quantity_skew_partitions
from federated.partitions.combined_skew import generate_controlled_combined_skew_partitions

__all__ = [
    "generate_controlled_quantity_skew_partitions",
    "generate_controlled_combined_skew_partitions",
]
