from __future__ import annotations

from io import BytesIO

import pandas as pd
import pytest

from eda_copilot.core.config import EDAValidationError
from eda_copilot.core.data_loading import infer_dataset_name, load_tabular_dataset


def test_load_tabular_dataset_reads_csv_metadata() -> None:
    uploaded = BytesIO(b"age,target\n25,0\n35,1\n")

    loaded = load_tabular_dataset(uploaded, "credit_sample.csv")

    assert loaded.dataset_name == "credit_sample"
    assert loaded.source_name == "credit_sample.csv"
    assert loaded.file_format == "csv"
    assert loaded.row_count == 2
    assert loaded.column_count == 2
    assert list(loaded.dataframe.columns) == ["age", "target"]
    assert loaded.memory_usage_bytes > 0


def test_load_tabular_dataset_reads_parquet_metadata() -> None:
    buffer = BytesIO()
    pd.DataFrame({"age": [25, 35], "target": [0, 1]}).to_parquet(buffer, index=False)

    loaded = load_tabular_dataset(buffer, "credit_sample.parquet")

    assert loaded.dataset_name == "credit_sample"
    assert loaded.file_format == "parquet"
    assert loaded.row_count == 2
    assert loaded.column_count == 2
    assert loaded.dataframe["target"].tolist() == [0, 1]


def test_load_tabular_dataset_rejects_unsupported_extension() -> None:
    with pytest.raises(EDAValidationError, match="Unsupported dataset file type '.txt'"):
        load_tabular_dataset(BytesIO(b"age,target\n25,0\n"), "credit_sample.txt")


def test_load_tabular_dataset_rejects_empty_csv() -> None:
    with pytest.raises(EDAValidationError, match="CSV file is empty"):
        load_tabular_dataset(BytesIO(b""), "empty.csv")


def test_load_tabular_dataset_rejects_header_only_csv() -> None:
    with pytest.raises(EDAValidationError, match="dataset is empty"):
        load_tabular_dataset(BytesIO(b"age,target\n"), "header_only.csv")


def test_load_tabular_dataset_rejects_malformed_csv() -> None:
    with pytest.raises(EDAValidationError, match="CSV parser found invalid structure"):
        load_tabular_dataset(BytesIO(b'age,target\n"25,0\n'), "malformed.csv")


def test_infer_dataset_name_falls_back_for_blank_stem() -> None:
    assert infer_dataset_name(".csv") == "dataset"
