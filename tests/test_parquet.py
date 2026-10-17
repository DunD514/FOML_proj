"""Unit tests for PyArrow schema validation and Parquet round-trip operations."""

import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

from src.storage.schema import get_inspection_telemetry_schema
from src.storage.parquet_writer import write_inspection_log, append_inspection_batch
from src.storage.parquet_reader import read_inspection_log, get_inspection_summary


def test_schema_definition():
    """Verify PyArrow schema completeness and metadata."""
    schema = get_inspection_telemetry_schema()
    assert isinstance(schema, pa.Schema)
    assert "inspection_id" in schema.names
    assert "confidence_score" in schema.names
    assert "drone_x" in schema.names
    assert schema.metadata is not None


def test_parquet_write_and_read(tmp_path: Path):
    """Test serializing inspection records to Parquet and reading back with metrics."""
    test_file = tmp_path / "test_mission.parquet"

    raw_records = [
        {
            "inspection_id": "TEST_001",
            "step_id": 1,
            "timestamp": "2026-10-17T12:00:00",
            "drone_x": 5.0,
            "drone_y": 3.0,
            "drone_z": 1.5,
            "drone_heading_deg": 45.0,
            "measured_x": 5.02,
            "measured_y": 2.98,
            "measured_z": 1.51,
            "measured_heading_deg": 45.5,
            "zone_id": "Bow-Waterline",
            "camera_image_id": "sample_img_01",
            "dataset_source": "test_source",
            "ground_truth_defect": True,
            "predicted_defect_class": "suspected_corrosion",
            "confidence_score": 0.88,
            "defect_detected": True,
            "defect_bbox": [0.2, 0.3, 0.6, 0.7],
            "segmentation_mask_reference": "masks/ref_01.png",
            "simulated_visibility": 0.80,
            "simulated_turbidity": 0.20,
            "inspection_status": "requires_further_inspection",
            "coverage_pct": 12.5,
            "battery_pct": 98.2,
        }
    ]

    # Write
    written_path = write_inspection_log(raw_records, test_file)
    assert written_path.exists()

    # Read
    df = read_inspection_log(test_file)
    assert len(df) == 1
    assert df["inspection_id"].iloc[0] == "TEST_001"
    assert df["defect_detected"].iloc[0] == True
    assert df["defect_x_min"].iloc[0] == 0.2

    # Summary
    summary = get_inspection_summary(test_file)
    assert summary["total_steps"] == 1
    assert summary["total_suspected_defects"] == 1


def test_parquet_empty_handling(tmp_path: Path):
    """Test graceful handling of empty record lists."""
    test_file = tmp_path / "empty.parquet"
    write_inspection_log([], test_file)

    assert test_file.exists()
    df = read_inspection_log(test_file)
    assert len(df) == 0
    assert "inspection_id" in df.columns
