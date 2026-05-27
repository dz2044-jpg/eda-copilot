from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal

import pandas as pd

from eda_copilot.core.config import EDAValidationError


DatasetFileFormat = Literal["csv", "parquet"]


@dataclass(frozen=True)
class LoadedDataset:
    """Validated tabular dataset loaded from a supported local file."""

    dataframe: pd.DataFrame
    dataset_name: str
    source_name: str
    file_format: DatasetFileFormat
    row_count: int
    column_count: int
    memory_usage_bytes: int


def load_tabular_dataset(file_obj: BinaryIO, source_name: str) -> LoadedDataset:
    """Load a CSV or Parquet file-like object into a validated dataframe.

    Args:
        file_obj: Binary file-like object from an upload or local caller.
        source_name: Source filename used to infer the dataset label and format.

    Returns:
        LoadedDataset with the dataframe and lightweight load metadata.

    Raises:
        EDAValidationError: If the file type is unsupported or cannot be loaded.
    """

    file_format = _infer_file_format(source_name)
    _seek_to_start(file_obj)
    if file_format == "csv":
        df = _read_csv(file_obj, source_name)
    else:
        df = _read_parquet(file_obj, source_name)
    _validate_loaded_dataframe(df)
    return LoadedDataset(
        dataframe=df,
        dataset_name=infer_dataset_name(source_name),
        source_name=source_name,
        file_format=file_format,
        row_count=int(len(df)),
        column_count=int(len(df.columns)),
        memory_usage_bytes=int(df.memory_usage(deep=True).sum()),
    )


def infer_dataset_name(source_name: str) -> str:
    """Infer a stable dataset label from a filename or source label."""

    stem = Path(source_name).stem.strip()
    if not stem or stem.startswith("."):
        return "dataset"
    return stem


def _infer_file_format(source_name: str) -> DatasetFileFormat:
    suffix = Path(source_name).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".parquet":
        return "parquet"
    label = suffix or "unknown"
    raise EDAValidationError(
        f"Unsupported dataset file type '{label}'. Upload a CSV or Parquet file."
    )


def _read_csv(file_obj: BinaryIO, source_name: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_obj)
    except pd.errors.EmptyDataError as exc:
        raise EDAValidationError(
            f"Could not load '{source_name}' because the CSV file is empty."
        ) from exc
    except pd.errors.ParserError as exc:
        raise EDAValidationError(
            f"Could not load '{source_name}' because the CSV parser found invalid structure."
        ) from exc
    except UnicodeDecodeError as exc:
        raise EDAValidationError(
            f"Could not load '{source_name}' because the CSV text encoding is not supported."
        ) from exc
    except ValueError as exc:
        raise EDAValidationError(f"Could not load '{source_name}' as CSV: {exc}") from exc


def _read_parquet(file_obj: BinaryIO, source_name: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(file_obj)
    except Exception as exc:
        raise EDAValidationError(f"Could not load '{source_name}' as Parquet: {exc}") from exc


def _validate_loaded_dataframe(df: pd.DataFrame) -> None:
    if len(df.columns) == 0:
        raise EDAValidationError("The dataset has no columns.")
    if len(df) == 0:
        raise EDAValidationError("The dataset is empty. Upload a file with at least one row.")


def _seek_to_start(file_obj: BinaryIO) -> None:
    try:
        file_obj.seek(0)
    except (AttributeError, OSError):
        return
