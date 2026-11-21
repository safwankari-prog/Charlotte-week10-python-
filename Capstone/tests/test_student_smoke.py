"""Student smoke checks. Add meaningful tests of edge cases yourself."""
from pathlib import Path
import pandas as pd
from src.production_pipeline import EXPECTED_COLUMNS, ingest_and_validate, repair_and_align, summarize_daily

DATA = Path(__file__).resolve().parents[1] / "data" / "historical_production_capstone.csv"

def test_source_columns():
    assert list(pd.read_csv(DATA, nrows=0).columns) == EXPECTED_COLUMNS

def test_ingestion_returns_two_dataframes():
    accepted, rejected = ingest_and_validate(DATA)
    assert isinstance(accepted, pd.DataFrame)
    assert isinstance(rejected, pd.DataFrame)

def test_aligned_and_summary_shapes():
    accepted, _ = ingest_and_validate(DATA)
    cleaned, report = repair_and_align(accepted)
    daily = summarize_daily(cleaned)
    assert len(cleaned) == 48  # 3 wells x 16 six-hour slots
    assert len(daily) == 12  # 3 wells x 4 days
    assert isinstance(report, pd.DataFrame)
