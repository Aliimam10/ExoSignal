"""Small data containers used by the data-pipeline stage."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class ProcessedSector:
    """One independently processed TESS sector."""

    sector: int
    data: pd.DataFrame
    diagnostics: dict[str, object]

