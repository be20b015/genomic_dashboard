"""
High-throughput, memory-safe FASTA/FASTQ parsing.

Design goals
------------
* Never write the uploaded file to disk — everything happens in an
  in-memory buffer (io.StringIO / io.BytesIO) so multi-GB uploads don't
  fill up ephemeral container storage.
* Stream records one at a time (generator) instead of loading the whole
  file into a list, so peak memory stays roughly constant regardless of
  file size.
* Use Biopython's low-level SimpleFastaParser / FastqGeneralIterator,
  which are much faster and lighter than SeqIO.parse() for large files
  because they skip building full SeqRecord objects until needed.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterator, Optional

from Bio.SeqIO.FastaIO import SimpleFastaParser
from Bio.SeqIO.QualityIO import FastqGeneralIterator


@dataclass
class SequenceRecord:
    id: str
    sequence: str
    quality: Optional[str] = None

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def gc_content(self) -> float:
        if not self.sequence:
            return 0.0
        seq = self.sequence.upper()
        gc = seq.count("G") + seq.count("C")
        return round(100 * gc / len(seq), 2)


def _sniff_format(text_head: str) -> str:
    stripped = text_head.lstrip()
    if stripped.startswith(">"):
        return "fasta"
    if stripped.startswith("@"):
        return "fastq"
    raise ValueError(
        "Unrecognized sequence format — file must start with '>' (FASTA) "
        "or '@' (FASTQ)."
    )


def iter_records_from_bytes(raw_bytes: bytes) -> Iterator[SequenceRecord]:
    """
    Stream SequenceRecord objects from raw uploaded bytes without ever
    touching disk. `raw_bytes` typically comes from Streamlit's
    UploadedFile.getvalue(), which itself is already buffered in RAM.
    """
    head = raw_bytes[:1024].decode("utf-8", errors="ignore")
    fmt = _sniff_format(head)

    text_buffer = io.StringIO(raw_bytes.decode("utf-8", errors="ignore"))

    if fmt == "fasta":
        for title, seq in SimpleFastaParser(text_buffer):
            record_id = title.split(None, 1)[0]
            yield SequenceRecord(id=record_id, sequence=seq)
    else:
        for title, seq, qual in FastqGeneralIterator(text_buffer):
            record_id = title.split(None, 1)[0]
            yield SequenceRecord(id=record_id, sequence=seq, quality=qual)


def stream_with_progress(raw_bytes: bytes, progress_callback=None, report_every: int = 500):
    """
    Wraps iter_records_from_bytes and invokes progress_callback(n_records,
    approx_bytes_consumed) periodically, so a Streamlit progress bar /
    status text can update during long parses without materializing the
    full record list first.
    """
    total_size = max(len(raw_bytes), 1)
    consumed = 0
    count = 0

    for record in iter_records_from_bytes(raw_bytes):
        count += 1
        # Approximate bytes consumed by this record (id + seq + quality + headers)
        consumed += len(record.id) + record.length + (len(record.quality) if record.quality else 0) + 4

        if progress_callback and count % report_every == 0:
            pct = min(consumed / total_size, 1.0)
            progress_callback(count, pct)

        yield record

    if progress_callback:
        progress_callback(count, 1.0)
