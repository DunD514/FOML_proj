"""Academic and engineering inspection report generator in PDF and Markdown formats."""

from typing import Dict, Any, List, Optional
import os
from pathlib import Path
from datetime import datetime
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from src.storage.parquet_reader import read_inspection_log, get_inspection_summary


DISCLAIMER_TEXT = (
    "<b>SCIENTIFIC & ENGINEERING LIMITATIONS:</b><br/>"
    "1. This automated inspection system identifies visible surface corrosion and cracking patterns "
    "using optical computer vision. Internal metallurgical stress, micro-fatigue, and subsurface cracks "
    "cannot be measured from RGB camera imagery alone.<br/>"
    "2. All model detections are classified as <i>'suspected defects'</i> with status <i>'requires further inspection'</i>. "
    "Confidence scores represent statistical model activation, not calibrated structural failure probabilities.<br/>"
    "3. Confirmatory structural integrity assessment requires certified marine surveyor evaluation and complementary "
    "non-destructive testing (NDT), such as ultrasonic thickness gauging (UT) or eddy current testing."
)


def generate_pdf_inspection_report(
    parquet_path: str | Path,
    output_pdf_path: str | Path
) -> Path:
    """Generate a formal, publication-ready PDF inspection report using ReportLab."""
    parquet_path = Path(parquet_path)
    output_pdf_path = Path(output_pdf_path)
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    summary = get_inspection_summary(parquet_path)
    df = read_inspection_log(parquet_path)
    defects_df = df[df["defect_detected"] == True]

    doc = SimpleDocTemplate(
        str(output_pdf_path),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1e3d59"),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#555555"),
        spaceAfter=15
    )
    heading2_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#17252a"),
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#333333")
    )
    disclaimer_style = ParagraphStyle(
        "DisclaimerCustom",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#7a1c1c")
    )

    story = []

    # Title & Header
    story.append(Paragraph("AUV / ROV SHIP HULL STRUCTURAL INSPECTION REPORT", title_style))
    story.append(Paragraph(
        f"Autonomous Robotic NDT Survey &bull; Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &bull; Mission ID: {summary['inspection_id']}",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1e3d59"), spaceAfter=12))

    # Executive KPI Summary Table
    story.append(Paragraph("1. Executive Mission Summary", heading2_style))
    kpi_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["Inspection ID", summary["inspection_id"], "Start Timestamp", summary["start_time"][:19]],
        ["Hull Surface Inspected", f"{summary['final_coverage_pct']}%", "Total Logged Steps", str(summary["total_steps"])],
        ["Suspected Defects Found", str(summary["total_suspected_defects"]), "Mean Detection Confidence", f"{summary['mean_confidence']*100:.1f}%"],
        ["Water Visibility Range", f"{summary['min_visibility']:.2f} - {summary['max_visibility']:.2f}", "ROV Battery Remaining", f"{summary['battery_remaining_pct']}%"]
    ]
    kpi_table = Table(kpi_data, colWidths=[130, 130, 130, 140])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3d59")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 12))

    # Suspected Defects Table
    story.append(Paragraph("2. Suspected Hull Defects Detected", heading2_style))
    if len(defects_df) == 0:
        story.append(Paragraph("No suspected defects were identified exceeding the detection confidence threshold.", body_style))
    else:
        defect_data = [["Step", "Zone ID", "Pose (X, Y) [m]", "Defect Category", "Confidence", "Inspection Status"]]
        for _, row in defects_df.head(20).iterrows():
            defect_data.append([
                str(row["step_id"]),
                str(row["zone_id"]),
                f"({row['drone_x']:.1f}, {row['drone_y']:.1f})",
                str(row["predicted_defect_class"]).replace("suspected_", "").capitalize(),
                f"{row['confidence_score']*100:.1f}%",
                "Requires Inspection"
            ])

        defect_table = Table(defect_data, colWidths=[40, 120, 100, 100, 70, 100])
        defect_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#b83b5e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fff5f5")]),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(defect_table)

    story.append(Spacer(1, 14))

    # Scientific Disclaimer Section
    story.append(Paragraph("3. Scientific Limitations & Marine Engineering Disclaimer", heading2_style))
    disclaimer_box = Table([[Paragraph(DISCLAIMER_TEXT, disclaimer_style)]], colWidths=[530])
    disclaimer_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff3cd")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#ffeeba")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(disclaimer_box)

    doc.build(story)
    return output_pdf_path


def generate_markdown_inspection_report(
    parquet_path: str | Path,
    output_md_path: str | Path
) -> Path:
    """Generate a clean Markdown inspection summary report."""
    parquet_path = Path(parquet_path)
    output_md_path = Path(output_md_path)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)

    summary = get_inspection_summary(parquet_path)
    df = read_inspection_log(parquet_path)
    defects_df = df[df["defect_detected"] == True]

    lines = [
        f"# Underwater ROV Hull Inspection Report",
        f"**Inspection Mission ID:** `{summary['inspection_id']}`  ",
        f"**Timestamp:** {summary['start_time']} &mdash; {summary['end_time']}  ",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "---",
        "## 1. Executive Summary",
        f"- **Hull Surface Inspected:** {summary['final_coverage_pct']}%",
        f"- **Total Inspection Steps:** {summary['total_steps']}",
        f"- **Suspected Defects Identified:** {summary['total_suspected_defects']}",
        f"- **Mean Defect Confidence:** {summary['mean_confidence']*100:.1f}%",
        f"- **Simulated Visibility Range:** {summary['min_visibility']:.2f} &ndash; {summary['max_visibility']:.2f}",
        f"- **ROV Remaining Battery:** {summary['battery_remaining_pct']}%\n",
        "## 2. Defect Detection Breakdown by Hull Zone",
    ]

    for zone, count in summary.get("defects_by_zone", {}).items():
        lines.append(f"- **Zone `{zone}`:** {count} suspected defect detections")

    lines.append("\n## 3. Detected Defect Log (First 15 Observations)")
    lines.append("| Step | Hull Zone | Coordinates (X, Y) [m] | Defect Category | Confidence | Status |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

    for _, row in defects_df.head(15).iterrows():
        cat = str(row["predicted_defect_class"]).replace("suspected_", "").capitalize()
        lines.append(
            f"| {row['step_id']} | {row['zone_id']} | ({row['drone_x']:.1f}, {row['drone_y']:.1f}) | {cat} | {row['confidence_score']*100:.1f}% | Requires Inspection |"
        )

    lines.append("\n## 4. Scientific Limitations & Marine Engineering Disclaimer")
    lines.append(
        "> **IMPORTANT:** Visible surface cracks and corrosion are detected using image-based computer vision. "
        "Internal stress, metal fatigue, and subsurface cracks cannot be determined from optical imagery alone "
        "and require ultrasonic thickness testing or strain gauge analysis. Detections are flagged as 'suspected defects' "
        "requiring verification by a certified marine surveyor."
    )

    content = "\n".join(lines)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_md_path
