"""Central Node / Aggregator for multi-cluster training using Flower with FedMom strategy.

This package contains:
  - strategy.py: FedMom strategy (Huo et al., 2020) for Flower.
  - server.py:   Flower server entrypoint running on open port 29511.
"""

from __future__ import annotations

from .strategy import FedMom

__all__ = ["FedMom"]
