"""Week 10 Class 5 capstone: implement the production-data pipeline.

Complete these functions independently. Keep the provided function names and signatures
so the supplied smoke tests can locate your work.
"""
from pathlib import Path
import pandas as pd
import numpy as np

EXPECTED_COLUMNS = ["well_id", "timestamp", "oil_bbl", "water_bbl", "gas_mscf", "pressure_psi", "status", "ingested_at"]


def ingest_and_validate(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (valid_timestamp_rows, rejected_timestamp_rows), including a rejection_reason.

    Never silently discard invalid timestamps. Validate required source columns.
    """
    df = pd.read_csv(path)
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    parsed_timestamp = pd.to_datetime(
        df["timestamp"],
        format="%Y-%m-%d %H:%M",
        errors="coerce"
    )

    invalid_timestamp_mask = parsed_timestamp.isna()
    rejected_rows = df.loc[invalid_timestamp_mask].copy()
    rejected_rows["rejection_reason"] = "Invalid timestamp"

    valid_rows = df.loc[~invalid_timestamp_mask].copy()
    valid_rows["timestamp"] = parsed_timestamp.loc[~invalid_timestamp_mask]

    return valid_rows, rejected_rows

def repair_and_align(valid_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (aligned_cleaned_rows, quality_report) for all wells, six-hour cadence.

    Duplicates: retain the latest ingested_at per well/timestamp. Negative values
    and nonnumeric numeric fields become missing. Do not impute production values.
    Pressure: interpolate at most ONE internal missing slot (no extrapolation).
    """
    working_rows = valid_rows.copy()
    working_rows = working_rows.sort_values("ingested_at")

    duplicate_count = working_rows.duplicated(
        subset=["well_id", "timestamp"]
    ).sum()

    working_rows = working_rows.drop_duplicates(
        subset=["well_id", "timestamp"],
        keep="last"
    )

    numeric_columns = ["oil_bbl", "water_bbl", "gas_mscf", "pressure_psi"]

    for column in numeric_columns:
        working_rows[column] = pd.to_numeric(
            working_rows[column],
            errors="coerce"
        )
        working_rows[column] = np.where(
            working_rows[column] < 0,
            np.nan,
            working_rows[column]
        )

    aligned_rows = []

    for well_id, well_data in working_rows.groupby("well_id"):
        well_data = well_data.set_index("timestamp").sort_index()

        expected_index = pd.date_range(
            start=well_data.index.min(),
            end=well_data.index.max(),
            freq="6h"
        )

        well_data["inserted_slot"] = False
        well_data = well_data.reindex(expected_index)
        well_data["inserted_slot"] = well_data["inserted_slot"].isna()

        well_data["pressure_psi"] = well_data["pressure_psi"].interpolate(
            method="time",
            limit=1,
            limit_area="inside"
        )

        well_data["well_id"] = well_id
        well_data = well_data.rename_axis("timestamp").reset_index()
        aligned_rows.append(well_data)

    aligned_cleaned_rows = pd.concat(aligned_rows, ignore_index=True)

    aligned_cleaned_rows["liquid_bbl"] = (
        aligned_cleaned_rows["oil_bbl"] + aligned_cleaned_rows["water_bbl"]
    )

    valid_liquid_mask = (
        aligned_cleaned_rows["oil_bbl"].notna()
        & aligned_cleaned_rows["water_bbl"].notna()
        & (aligned_cleaned_rows["liquid_bbl"] > 0)
    )

    aligned_cleaned_rows["water_cut_pct"] = np.nan

    aligned_cleaned_rows.loc[valid_liquid_mask, "water_cut_pct"] = (
        aligned_cleaned_rows.loc[valid_liquid_mask, "water_bbl"]
        / aligned_cleaned_rows.loc[valid_liquid_mask, "liquid_bbl"]
        * 100
    )

    quality_report = pd.DataFrame([{
        "duplicate_rows_removed": int(duplicate_count),
        "inserted_slots": int(aligned_cleaned_rows["inserted_slot"].sum()),
        "missing_oil_bbl": int(aligned_cleaned_rows["oil_bbl"].isna().sum()),
        "missing_water_bbl": int(aligned_cleaned_rows["water_bbl"].isna().sum()),
        "missing_gas_mscf": int(aligned_cleaned_rows["gas_mscf"].isna().sum()),
        "missing_pressure_psi": int(aligned_cleaned_rows["pressure_psi"].isna().sum())
    }])

    return aligned_cleaned_rows, quality_report

def summarize_daily(cleaned_rows: pd.DataFrame) -> pd.DataFrame:
    """Return daily oil/water/gas totals by well and date with completeness counts.

    Use min_count=1 for numerical production sums; no invented zeros.
    """
    daily_rows = cleaned_rows.copy()

    daily_rows["timestamp"] = pd.to_datetime(daily_rows["timestamp"])
    daily_rows["date"] = daily_rows["timestamp"].dt.date

    daily_summary = (
        daily_rows.groupby(["well_id", "date"], as_index=False)
        .agg(
            oil_bbl=("oil_bbl", lambda values: values.sum(min_count=1)),
            water_bbl=("water_bbl", lambda values: values.sum(min_count=1)),
            gas_mscf=("gas_mscf", lambda values: values.sum(min_count=1)),
            expected_slots=("timestamp", "size"),
            oil_valid_readings=("oil_bbl", "count"),
            water_valid_readings=("water_bbl", "count"),
            gas_valid_readings=("gas_mscf", "count")
        )
    )

    daily_summary["oil_complete"] = (
        daily_summary["oil_valid_readings"]
        == daily_summary["expected_slots"]
    )

    daily_summary["water_complete"] = (
        daily_summary["water_valid_readings"]
        == daily_summary["expected_slots"]
    )

    daily_summary["gas_complete"] = (
        daily_summary["gas_valid_readings"]
        == daily_summary["expected_slots"]
    )

    return daily_summary


def run_pipeline(input_csv: str | Path, output_dir: str | Path) -> None:
    """Create cleaned_production.csv, data_quality_report.csv, daily_production_summary.csv,
    and rejected_source_records.csv; make paths reproducible.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    valid_rows, rejected_rows = ingest_and_validate(input_csv)

    cleaned_rows, quality_report = repair_and_align(valid_rows)

    daily_summary = summarize_daily(cleaned_rows)

    cleaned_rows.to_csv(
        output_path / "cleaned_production.csv",
        index=False
    )

    quality_report.to_csv(
        output_path / "data_quality_report.csv",
        index=False
    )

    daily_summary.to_csv(
        output_path / "daily_production_summary.csv",
        index=False
    )

    rejected_rows.to_csv(
        output_path / "rejected_source_records.csv",
        index=False
    )