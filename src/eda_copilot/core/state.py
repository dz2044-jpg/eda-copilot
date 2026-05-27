from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EDAResult:
    """In-memory result returned by the workflow."""

    evidence_packet: dict[str, Any]
    markdown_report: str
    artifact_dir: Path | None = None


@dataclass(frozen=True)
class ArtifactFile:
    """File saved during an EDA run."""

    label: str
    path: Path
    kind: str
