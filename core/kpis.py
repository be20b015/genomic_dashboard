"""
KPI aggregation over a streamed collection of SequenceRecord objects.

Kept O(1) in memory per-record: we maintain running sums/counters rather
than storing every sequence, so this scales to millions of records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .parser import SequenceRecord


@dataclass
class RunningKPIs:
    n_records: int = 0
    total_length: int = 0
    total_gc: int = 0
    min_length: int = None
    max_length: int = None
    length_samples: List[int] = field(default_factory=list)  # reservoir for histogram
    gc_samples: List[float] = field(default_factory=list)
    ambiguous_base_count: int = 0  # N, or non-ACGT/U bases — proxy for "anomalies"
    _max_samples: int = 20000

    def update(self, record: SequenceRecord) -> None:
        self.n_records += 1
        self.total_length += record.length
        seq_upper = record.sequence.upper()
        gc = seq_upper.count("G") + seq_upper.count("C")
        self.total_gc += gc

        if self.min_length is None or record.length < self.min_length:
            self.min_length = record.length
        if self.max_length is None or record.length > self.max_length:
            self.max_length = record.length

        ambiguous = sum(1 for b in seq_upper if b not in "ACGTU")
        self.ambiguous_base_count += ambiguous

        # Reservoir-style cap so histogram data doesn't grow unbounded on huge files
        if len(self.length_samples) < self._max_samples:
            self.length_samples.append(record.length)
            self.gc_samples.append(record.gc_content)

    @property
    def mean_length(self) -> float:
        return round(self.total_length / self.n_records, 1) if self.n_records else 0.0

    @property
    def overall_gc_pct(self) -> float:
        return round(100 * self.total_gc / self.total_length, 2) if self.total_length else 0.0

    @property
    def ambiguous_pct(self) -> float:
        return round(100 * self.ambiguous_base_count / self.total_length, 4) if self.total_length else 0.0

    def summary_dict(self) -> dict:
        return {
            "records": self.n_records,
            "total_bases": self.total_length,
            "mean_length": self.mean_length,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "overall_gc_pct": self.overall_gc_pct,
            "ambiguous_base_pct": self.ambiguous_pct,
        }
