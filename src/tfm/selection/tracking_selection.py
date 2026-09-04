from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_selection_table(
    csv_path: str | Path,
    required_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load and validate the tracking-selection CSV table.

    Parameters
    ----------
    csv_path : str | Path
        Path to the CSV file containing the offline best configuration per
        target signature.
    required_columns : list[str] | None, optional
        List of columns that must exist in the CSV. If None, the default
        minimal set required for lookup is used.

    Returns
    -------
    pd.DataFrame
        Clean DataFrame restricted to the required columns, without duplicated
        rows.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Selection CSV not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)

    if required_columns is None:
        required_columns = [
            "target_signature",
            "recommended_policy",
            "recommended_tracker",
        ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            "Selection CSV is missing required columns: "
            f"{missing_columns}. Available columns: {list(df.columns)}"
        )

    clean_df = df[required_columns].copy().drop_duplicates().reset_index(drop=True)

    if clean_df.empty:
        raise ValueError("Selection CSV is empty after cleaning.")

    return clean_df


def build_selection_lookup(
    selection_df: pd.DataFrame,
) -> dict[str, tuple[str, str]]:
    """
    Build a lookup dictionary from the cleaned selection table.

    The key is:
        target_signature

    The value is:
        (recommended_policy, recommended_tracker)

    Parameters
    ----------
    selection_df : pd.DataFrame
        Clean selection DataFrame.

    Returns
    -------
    dict[str, tuple[str, str]]
        Lookup dictionary for fast configuration retrieval.
    """
    required_columns = {
        "target_signature",
        "recommended_policy",
        "recommended_tracker",
    }

    missing_columns = required_columns.difference(selection_df.columns)
    if missing_columns:
        raise ValueError(
            "Selection DataFrame is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    duplicated_mask = selection_df.duplicated(
        subset=["target_signature"],
        keep=False,
    )

    if duplicated_mask.any():
        duplicated_rows = selection_df.loc[
            duplicated_mask,
            ["target_signature"],
        ]
        raise ValueError(
            "Ambiguous selection table: duplicated target_signature keys found.\n"
            f"{duplicated_rows.to_string(index=False)}"
        )

    lookup: dict[str, tuple[str, str]] = {}

    for _, row in selection_df.iterrows():
        key = str(row["target_signature"])
        value = (
            str(row["recommended_policy"]),
            str(row["recommended_tracker"]),
        )
        lookup[key] = value

    if not lookup:
        raise ValueError("Selection lookup is empty.")

    return lookup


def load_selection_lookup(
    csv_path: str | Path,
    required_columns: list[str] | None = None,
) -> dict[str, tuple[str, str]]:
    """
    Load the CSV and directly build the lookup dictionary.

    Parameters
    ----------
    csv_path : str | Path
        Path to the CSV file.
    required_columns : list[str] | None, optional
        Columns to keep and validate. If None, the default minimal set is used.

    Returns
    -------
    dict[str, tuple[str, str]]
        Lookup dictionary mapping:
            target_signature
        to:
            (recommended_policy, recommended_tracker)
    """
    selection_df = load_selection_table(
        csv_path=csv_path,
        required_columns=required_columns,
    )
    return build_selection_lookup(selection_df)


def select_tracking_configuration(
    target_signature: str,
    selection_lookup: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    """
    Select the recommended tracking configuration from a target signature.

    Parameters
    ----------
    target_signature : str
        Canonical target signature, e.g.:
            "0x_none"
            "1x_dummy"
            "2x_drone + 1x_truck"
    selection_lookup : dict[str, tuple[str, str]]
        Lookup table mapping:
            target_signature
        to:
            (recommended_policy, recommended_tracker)

    Returns
    -------
    dict[str, Any]
        Dictionary with:
            - target_signature
            - recommended_policy
            - recommended_tracker
    """
    if not isinstance(selection_lookup, dict) or len(selection_lookup) == 0:
        raise ValueError("selection_lookup must be a non-empty dictionary.")

    key = str(target_signature)

    if key not in selection_lookup:
        raise KeyError(f"Target signature '{target_signature}' not found in lookup.")

    recommended_policy, recommended_tracker = selection_lookup[key]

    return {
        "target_signature": key,
        "recommended_policy": recommended_policy,
        "recommended_tracker": recommended_tracker,
    }