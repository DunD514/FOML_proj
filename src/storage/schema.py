"""Strongly typed PyArrow schema definition for underwater inspection telemetry and defect logs."""

import pyarrow as pa


def get_inspection_telemetry_schema() -> pa.Schema:
    """Return the official Apache Arrow schema for ROV hull inspection records.

    Field types and nullability are strictly defined to guarantee consistency across
    parquet files and analytical queries.
    """
    fields = [
        # Mission identifiers
        pa.field("inspection_id", pa.string(), nullable=False),
        pa.field("step_id", pa.int64(), nullable=False),
        pa.field("timestamp", pa.string(), nullable=False),

        # Ground-truth drone coordinates
        pa.field("drone_x", pa.float64(), nullable=False),
        pa.field("drone_y", pa.float64(), nullable=False),
        pa.field("drone_z", pa.float64(), nullable=False),
        pa.field("drone_heading_deg", pa.float64(), nullable=False),

        # Simulated sensor measurements (with acoustic/DVL noise)
        pa.field("measured_x", pa.float64(), nullable=False),
        pa.field("measured_y", pa.float64(), nullable=False),
        pa.field("measured_z", pa.float64(), nullable=False),
        pa.field("measured_heading_deg", pa.float64(), nullable=False),

        # Hull spatial segmentation
        pa.field("zone_id", pa.string(), nullable=False),

        # Image observation metadata (references only, no bulk blob storage)
        pa.field("camera_image_id", pa.string(), nullable=False),
        pa.field("dataset_source", pa.string(), nullable=False),
        pa.field("ground_truth_defect", pa.bool_(), nullable=False),

        # ML model predictions
        pa.field("predicted_defect_class", pa.string(), nullable=False),
        pa.field("confidence_score", pa.float64(), nullable=False),
        pa.field("defect_detected", pa.bool_(), nullable=False),

        # Normalized defect bounding box [0.0 - 1.0] (nullable if no defect)
        pa.field("defect_x_min", pa.float64(), nullable=True),
        pa.field("defect_y_min", pa.float64(), nullable=True),
        pa.field("defect_x_max", pa.float64(), nullable=True),
        pa.field("defect_y_max", pa.float64(), nullable=True),

        # Segmentation mask file reference (nullable)
        pa.field("segmentation_mask_reference", pa.string(), nullable=True),

        # Environmental conditions
        pa.field("simulated_visibility", pa.float64(), nullable=False),
        pa.field("simulated_turbidity", pa.float64(), nullable=False),

        # Operational status and mission telemetry
        pa.field("inspection_status", pa.string(), nullable=False),
        pa.field("coverage_pct", pa.float64(), nullable=False),
        pa.field("battery_pct", pa.float64(), nullable=False),
    ]

    metadata = {
        b"project": b"AI-Powered Underwater Ship Hull Inspection Drone",
        b"schema_version": b"1.0.0",
        b"created_by": b"FOML Machine Learning Pipeline",
    }

    return pa.schema(fields, metadata=metadata)
