from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st


def render_optional_pygwalker(df: pd.DataFrame, dataset_name: str) -> None:
    """Render the optional PyGWalker explorer when the integration is installed."""

    try:
        import pygwalker  # noqa: F401
    except ImportError:
        st.info("Optional PyGWalker explorer is not installed.")
        return

    if not st.checkbox("Open PyGWalker explorer"):
        return

    spec_dir = Path("artifacts/runs/visual_specs")
    try:
        spec_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        st.error(f"Could not prepare the visual explorer workspace: {exc}")
        return

    spec_path = spec_dir / f"{_dataset_key(df, dataset_name)}.json"
    renderer = _pygwalker_renderer(_dataset_key(df, dataset_name), df, str(spec_path))
    renderer.explorer()


@st.cache_resource(show_spinner=False)
def _pygwalker_renderer(dataset_key: str, _df: pd.DataFrame, spec_path: str):
    from pygwalker.api.streamlit import StreamlitRenderer

    return StreamlitRenderer(_df, spec=spec_path, spec_io_mode="rw")


def _dataset_key(df: pd.DataFrame, dataset_name: str) -> str:
    columns = "|".join(str(column) for column in df.columns)
    digest = hashlib.sha256()
    digest.update(f"{dataset_name}|{len(df)}|{len(df.columns)}|{columns}".encode("utf-8"))
    try:
        digest.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    except (TypeError, ValueError):
        digest.update(pd.util.hash_pandas_object(df.astype(str), index=True).values.tobytes())
    digest_text = digest.hexdigest()[:12]
    return f"{dataset_name}_{len(df)}x{len(df.columns)}_{digest_text}"
