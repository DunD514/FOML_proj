"""Parquet writer utility for atomic logging of inspection batches."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import pyarrow as pa
import pyarrow.parquet as pq

from src.storage.schema import get_inspection_telemetry_schema


def format_records_for_schema(raw_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format and typecast raw simulation logs into exact schema fields."""
    formatted = []
    for r in raw_records:
        # Extract bounding box coordinates if present
        x_min, y_min, x_max, y_max = None, None, None, None
        bbox_val = r.get("defect_bbox")
        if bbox_val:
            if isinstance(bbox_val, str):
                try:
                    coords = json.loads(bbox_val)
                    if len(coords) == 4:
                        x_min, y_min, x_max, y_max = coords
                except Exception:
                    pass
            elif isinstance(bbox_val, (list, tuple)) and len(bbox_val) == 4:
                x_min, y_min, x_max, y_max = bbox_val

        rec = {
            "inspection_id": str(r["inspection_id"]),
            "step_id": int(r["step_id"]),
            "timestamp": str(r["timestamp"]),
            "drone_x": float(r["drone_x"]),
            "drone_y": float(r["drone_y"]),
            "drone_z": float(r["drone_z"]),
            "drone_heading_deg": float(r["drone_heading_deg"]),
            "measured_x": float(r["measured_x"]),
            "measured_y": float(r["measured_y"]),
            "measured_z": float(r["measured_z"]),
            "measured_heading_deg": float(r["measured_heading_deg"]),
            "zone_id": str(r["zone_id"]),
            "camera_image_id": str(r["camera_image_id"]),
            "dataset_source": str(r["dataset_source"]),
            "ground_truth_defect": bool(r["ground_truth_defect"]),
            "predicted_defect_class": str(r["predicted_defect_class"]),
            "confidence_score": float(r["confidence_score"]),
            "defect_detected": bool(r["defect_detected"]),
            "defect_x_min": float(x_min) if x_min is not None else None,
            "defect_y_min": float(y_min) if y_min is not None else None,
            "defect_x_max": float(x_max) if x_max is not None else None,
            "defect_y_max": float(y_max) if y_max is not None else None,
            "segmentation_mask_reference": str(r["segmentation_mask_reference"]) if r.get("segmentation_mask_reference") else None,
            "simulated_visibility": float(r["simulated_visibility"]),
            "simulated_turbidity": float(r["simulated_turbidity"]),
            "inspection_status": str(r["inspection_status"]),
            "coverage_pct": float(r["coverage_pct"]),
            "battery_pct": float(r["battery_pct"]),
        }
        formatted.append(rec)
    return formatted


def write_inspection_log(
    records: List[Dict[str, Any]],
    output_path: str | Path,
    compression: str = "snappy"
) -> Path:
    """Serialize inspection telemetry records to an Apache Parquet file using the official schema."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = get_inspection_telemetry_schema()

    if not records:
        # Create empty table with valid schema
        table = pa.Table.from_arrays([pa.array([], type=f.type) for f in schema], schema=schema)
    else:
        formatted = format_records_for_schema(records)
        # Build columnar arrays
        arrays = []
        for f in schema:
            col_data = [row[f.name] for row in formatted]
            arrays.append(pa.array(col_data, type=f.type))
        table = pa.Table.from_arrays(arrays, schema=schema)

    pq.write_table(table, str(output_path), compression=compression)
    return output_path


def append_inspection_batch(
    new_records: List[Dict[str, Any]],
    target_parquet: str | Path,
    compression: str = "snappy"
) -> Path:
    """Append a new batch of records to an existing Parquet file."""
    target_parquet = Path(target_parquet)
    if not target_parquet.exists():
        return write_inspection_log(new_records, target_parquet, compression=compression)

    existing_table = pq.read_table(str(target_parquet))
    existing_records = existing_table.to_pylist()
    combined_records = existing_records + new_records
    return write_inspection_log(combined_records, target_parquet, compression=compression)
