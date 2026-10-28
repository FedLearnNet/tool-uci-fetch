from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, List

import pandas as pd
from pydantic.dataclasses import dataclass
from pyfedappwrap.learning.run_runfig import AppConfig, AppInputConfig, AppOutputConfig
from pyfedappwrap.types.databaseadopter.base_importer import BaseExtractorAdopterAPP
from ucimlrepo import fetch_ucirepo


def _to_df(obj: Any, name: str) -> Optional[pd.DataFrame]:
    """Normalize DataFrame/Series to DataFrame. Return None if not tabular."""
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, pd.Series):
        return obj.to_frame(name=name)
    return None


def _collect_tabular_from_data(data_obj: Any) -> List[pd.DataFrame]:
    """
    Heuristically collect DataFrames/Series from dataset.data.
    We count them to warn if there's more than two.
    """
    tabular: List[pd.DataFrame] = []

    # Prefer common fields if present
    for attr in ("features", "targets", "ids"):
        if hasattr(data_obj, attr):
            df = _to_df(getattr(data_obj, attr), attr)
            if df is not None:
                tabular.append(df)

    # Also scan public attributes for extra tabular payloads
    for attr in dir(data_obj):
        if attr.startswith("_") or attr in {"features", "targets", "ids"}:
            continue
        try:
            val = getattr(data_obj, attr)
        except Exception:
            continue
        df = _to_df(val, attr)
        if df is not None:
            tabular.append(df)

    # Deduplicate by object id (avoid double counting)
    seen = set()
    unique = []
    for df in tabular:
        if id(df) in seen:
            continue
        seen.add(id(df))
        unique.append(df)
    return unique


def _merge_features_targets(features: Optional[pd.DataFrame],
                            targets: Optional[pd.DataFrame]) -> pd.DataFrame:
    if features is None and targets is None:
        raise ValueError("UCI dataset returned neither features nor targets.")
    if features is None:
        return targets.copy()
    if targets is None:
        return features.copy()

    overlap = set(features.columns).intersection(set(targets.columns))
    if overlap:
        targets = targets.copy()
        targets.columns = [f"target__{c}" if c in overlap else c for c in targets.columns]

    return pd.concat([features.reset_index(drop=True), targets.reset_index(drop=True)], axis=1)


@dataclass
class UciFetchToCsvAppConfig(AppConfig):
    dataset_id: int = 17  # default: Breast Cancer Wisconsin (Diagnostic)


@dataclass
class UciFetchToCsvAppInputConfig(AppInputConfig):
    # no inputs required
    pass


@dataclass
class UciFetchToCsvAppOutputConfig(AppOutputConfig):
    dataset: Path | None = None
    report: str | None = None


class UciFetchToCsvAPP(
    BaseExtractorAdopterAPP[
        UciFetchToCsvAppConfig, UciFetchToCsvAppInputConfig, UciFetchToCsvAppOutputConfig]
):
    """
    Fetch a UCI dataset by numeric id via ucimlrepo and export as a single CSV:
      - features + targets concatenated horizontally (one big table)
    Logs a warning if more than two tabular datasets are detected on dataset.data.
    """

    def run_adopter(self) -> UciFetchToCsvAppOutputConfig:
        dataset_id = int(self.config.dataset_id)
        self.logger.info(f"Fetching UCI dataset id={dataset_id} via ucimlrepo")

        ds = fetch_ucirepo(id=dataset_id)

        tabular_payloads = _collect_tabular_from_data(ds.data)
        if len(tabular_payloads) > 2:
            self.logger.warning(
                f"Dataset id={dataset_id} exposes {len(tabular_payloads)} tabular payloads on ds.data. "
                f"Only features+targets are supported; extras will be ignored."
            )

        X = _to_df(getattr(ds.data, "features", None), "features")
        y = _to_df(getattr(ds.data, "targets", None), "targets")

        merged = _merge_features_targets(X, y)

        out_path = self.save_df(merged, "uci_dataset"+str(dataset_id))

        # Minimal report
        meta_name = None
        try:
            meta_name = ds.metadata.get("name")
        except Exception:
            meta_name = None

        report = (
                f"Fetched UCI dataset id={dataset_id}"
                + (f" ({meta_name})" if meta_name else "")
                + f". Output rows={merged.shape[0]}, cols={merged.shape[1]}."
        )

        self.send_metric("rows", float(merged.shape[0]), 0)
        self.send_metric("cols", float(merged.shape[1]), 0)

        return UciFetchToCsvAppOutputConfig(dataset=out_path, report=report)
