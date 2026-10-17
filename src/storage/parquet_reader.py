"""Parquet reading and analytical query module for inspection logs."""

from typing import Dict, Any, List, Optional
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
import numpy as np


def read_inspection_log(file_path: str | Path) -> pd.DataFrame:
    """Read a Parquet inspection log file into a Pandas DataFrame."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet log file not found at: {file_path}")
    table = pq.read_table(str(file_path))
    return table.to_pandas()


def list_available_inspections(logs_dir: str | Path) -> List[Dict[str, Any]]:
    """Scan and list all recorded inspection logs with top-level metadata."""
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        return []

    inspections = []
    for p_file in sorted(logs_path.glob("*.parquet"), reverse=True):
        try:
            meta = pq.read_metadata(str(p_file))
            df = read_inspection_log(p_file)
            insp_id = df["inspection_id"].iloc[0] if len(df) > 0 else p_file.stem
            defects_found = int(df["defect_detected"].sum()) if "defect_detected" in df else 0
            max_cov = float(df["coverage_pct"].max()) if "coverage_pct" in df and len(df) > 0 else 0.0

            inspections.append({
                "inspection_id": insp_id,
                "file_path": str(p_file),
                "total_records": meta.num_rows,
                "defects_detected": defects_found,
                "final_coverage_pct": max_cov,
                "timestamp": df["timestamp"].iloc[0] if len(df) > 0 else "Unknown"
            })
        except Exception as e:
            print(f"Warning reading {p_file.name}: {e}")

    return inspections


def get_inspection_summary(file_path: str | Path) -> Dict[str, Any]:
    """Compute analytical summary metrics from a Parquet inspection log."""
    df = read_inspection_log(file_path)
    if len(df) == 0:
        return {"error": "Inspection log is empty"}

    defects_df = df[df["defect_detected"] == True]
    zone_defect_counts = defects_df["zone_id"].value_counts().to_dict() if len(defects_df) > 0 else {}
    class_counts = defects_df["predicted_defect_class"].value_counts().to_dict() if len(defects_df) > 0 else {}

    return {
        "inspection_id": str(df["inspection_id"].iloc[0]),
        "total_steps": int(len(df)),
        "start_time": str(df["timestamp"].iloc[0]),
        "end_time": str(df["timestamp"].iloc[-1]),
        "final_coverage_pct": round(float(df["coverage_pct"].max()), 2),
        "total_suspected_defects": int(len(defects_df)),
        "defects_by_zone": zone_defect_counts,
        "defects_by_class": class_counts,
        "mean_confidence": round(float(df["confidence_score"].mean()), 4),
        "min_visibility": round(float(df["simulated_visibility"].min()), 3),
        "max_visibility": round(float(df["simulated_visibility"].max()), 3),
        "battery_remaining_pct": round(float(df["battery_pct"].iloc[-1]), 1),
    }


def export_inspection_to_csv(parquet_path: str | Path, csv_output_path: str | Path) -> Path:
    """Export a Parquet inspection log to CSV format."""
    df = read_inspection_log(parquet_path)
    csv_path = Path(csv_output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path
