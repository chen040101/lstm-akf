from __future__ import annotations

from pathlib import Path

from lstm_akf.datasets.builder import JsonSampleDataset, SavedSampleDataset


def load_dataset(sample_path: str | Path, return_meta: bool = False, **kwargs):
    sample_path = Path(sample_path)
    if sample_path.suffix.lower() == ".pt":
        return SavedSampleDataset(sample_path, return_meta=return_meta, **kwargs)
    return JsonSampleDataset(sample_path, return_meta=return_meta, **kwargs)


__all__ = ["JsonSampleDataset", "SavedSampleDataset", "load_dataset"]
