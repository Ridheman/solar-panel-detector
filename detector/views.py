
from pathlib import Path
from uuid import uuid4

import cv2
import torch
import joblib
import numpy as np

from PIL import Image
from django.conf import settings
from django.shortcuts import render
from torchvision import models, transforms
from ultralytics import YOLO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image as PDFImage, Table, TableStyle
)
from datetime import datetime


# Project paths
BASE_DIR = Path(settings.BASE_DIR)

YOLO_PATH = BASE_DIR / "models" / "panel_segmenter.pt"

SVM_PATH = BASE_DIR / "models" / "solar_svm.pkl"
SCALER_PATH = BASE_DIR / "models" / "solar_scaler.pkl"
CNN_PATH = BASE_DIR / "models" / "mobilenetv2_features.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models only once
yolo = None
cnn = None
svm = None
scaler = None


def load_models():
    global yolo, cnn, svm, scaler

    if yolo is not None:
        return

    print("Loading SolarGuard models...")

    yolo = YOLO(str(YOLO_PATH))

    mobilenet = models.mobilenet_v2(weights=None)
    cnn = mobilenet.features

    cnn.load_state_dict(
        torch.load(
            CNN_PATH,
            map_location=DEVICE,
            weights_only=True
        )
    )

    cnn = cnn.to(DEVICE)
    cnn.eval()

    svm = joblib.load(SVM_PATH)
    scaler = joblib.load(SCALER_PATH)

    print("All models loaded successfully!")


# Same preprocessing used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def analyse_image(image_path, output_path):

    load_models()

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Could not read the uploaded image.")

    # YOLO panel detection
    results = yolo.predict(
        image,
        conf=0.25,
        verbose=False
    )

    result = results[0]
    output = image.copy()

    clean_count = 0
    dusty_count = 0

    panel_results = []

    boxes = result.boxes

    if boxes is not None:
        for i, box in enumerate(boxes):

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].cpu().numpy()
            )

            # Keep coordinates within image boundaries
            h, w = image.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # Convert panel crop into model input
            pil_image = Image.fromarray(
                cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            )

            tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

            # CNN feature extraction
            with torch.no_grad():
                features = cnn(tensor)

                features = torch.nn.functional.adaptive_avg_pool2d(
                    features, (1, 1)
                )

                features = torch.flatten(features, 1)

            features = features.cpu().numpy()

            # SVM classification
            features = scaler.transform(features)

            prediction = int(svm.predict(features)[0])

            # Absolute distance from the SVM decision boundary
            decision_score = float(
                np.abs(svm.decision_function(features)[0])
            )

            panel_results.append({
                "panel_id": i + 1,
                "status": "Dusty" if prediction == 1 else "Clean",
                "decision_score": round(decision_score, 4)
            })
            

            if prediction == 1:
                label = "Dusty"
                color = (0, 0, 255)
                dusty_count += 1
            else:
                label = "Clean"
                color = (0, 200, 0)
                clean_count += 1

            # Draw bounding box
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            # Readable label background
           # Draw a clean bounding box without overlapping text
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

# Small numbered marker
            cv2.circle(
                output,
                (x1 + 15, y1 + 15),
                13,
                color,
                -1
            )

            cv2.putText(
                output,
                str(i + 1),
                (x1 + 10, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2
            )

    total = clean_count + dusty_count

    # Save annotated result
    cv2.imwrite(str(output_path), output)

    dusty_percentage = (
        round((dusty_count / total) * 100, 2)
        if total else 0
    )

    return {
        "total": total,
        "clean": clean_count,
        "dusty": dusty_count,
        "dusty_percentage": dusty_percentage,
        "panel_results": panel_results
    }



def generate_pdf_report(input_path, output_path, report_path, summary):

    doc = SimpleDocTemplate(
        str(report_path),
        pagesize=A4,
        rightMargin=35,
        leftMargin=35,
        topMargin=35,
        bottomMargin=35
    )

    styles = getSampleStyleSheet()
    elements = []

    # Report heading
    elements.append(Paragraph(
        "SOLARGUARD",
        styles["Title"]
    ))

    elements.append(Paragraph(
        "Solar Panel Inspection Report",
        styles["Heading2"]
    ))

    elements.append(Paragraph(
        f"Inspection Date: {datetime.now().strftime('%d %B %Y, %H:%M')}",
        styles["Normal"]
    ))

    elements.append(Spacer(1, 20))

    # Summary table
    elements.append(Paragraph("Inspection Summary", styles["Heading2"]))

    summary_data = [
        ["Total Panels", "Clean Panels", "Dusty Panels", "Dusty %"],
        [
            str(summary["total"]),
            str(summary["clean"]),
            str(summary["dusty"]),
            f'{summary["dusty_percentage"]}%'
        ]
    ]

    summary_table = Table(summary_data, colWidths=[125] * 4)

    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#174D38")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("TEXTCOLOR", (0, 1), (0, 1), colors.HexColor("#172B25")),
        ("TEXTCOLOR", (1, 1), (1, 1), colors.HexColor("#16834D")),
        ("TEXTCOLOR", (2, 1), (2, 1), colors.HexColor("#D43B35")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("PADDING", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")
    ]))

    elements.append(summary_table)
    elements.append(Spacer(1, 15))

    # Visual distribution bar
    elements.append(Paragraph("Panel Classification Distribution", styles["Heading2"]))

    clean = summary["clean"]
    dusty = summary["dusty"]

    if clean + dusty > 0:
        bar = Table(
            [[f"Clean: {clean}", f"Dusty: {dusty}"]],
            colWidths=[
                max(1, 400 * clean / (clean + dusty)),
                max(1, 400 * dusty / (clean + dusty))
            ]
        )

        bar.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#16834D")),
            ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#D43B35")),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("PADDING", (0, 0), (-1, -1), 9)
        ]))

        elements.append(bar)

    elements.append(Spacer(1, 20))

    # Images
    for path, heading in [
        (input_path, "Original Image"),
        (output_path, "Analysed Image")
    ]:
        elements.append(Paragraph(heading, styles["Heading2"]))

        img = Image.open(path)
        width, height = img.size

        ratio = min(500 / width, 280 / height)

        elements.append(PDFImage(
            str(path),
            width=width * ratio,
            height=height * ratio
        ))

        elements.append(Spacer(1, 15))

    # Panel details
    elements.append(Paragraph(
        "Individual Panel Classification",
        styles["Heading2"]
    ))

    panel_data = [["Panel ID", "Classification", "SVM Decision Score"]]

    for panel in summary["panel_results"]:
        panel_data.append([
            f'Panel {panel["panel_id"]}',
            panel["status"],
            str(panel.get("decision_score", "N/A"))
        ])

    panel_table = Table(
        panel_data,
        colWidths=[130, 130, 160],
        repeatRows=1
    )

    panel_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#174D38")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.white, colors.HexColor("#F1F6F3")
        ])
    ]))

    elements.append(panel_table)
    elements.append(Spacer(1, 15))

    elements.append(Paragraph(
        "Methodology: Solar panel detection is performed using YOLOv8 "
        "segmentation model predictions. Classification uses MobileNetV2 "
        "feature extraction and an SVM classifier. SVM decision scores "
        "are uncalibrated margins, not probabilities. Dusty percentage "
        "represents the proportion of panels classified as Dusty, not "
        "physical dust coverage.",
        styles["Normal"]
    ))

    doc.build(elements)


def home(request):

    context = {}

    if request.method == "POST":

        uploaded_file = request.FILES.get("solar_image")

        if not uploaded_file:
            context["error"] = "Please select an image first."
            return render(request, "detector/home.html", context)

        try:
            # Create folders
            upload_dir = Path(settings.MEDIA_ROOT) / "uploads"
            result_dir = Path(settings.MEDIA_ROOT) / "results"

            upload_dir.mkdir(parents=True, exist_ok=True)
            result_dir.mkdir(parents=True, exist_ok=True)

            # Unique filenames
            filename = f"{uuid4().hex}.jpg"

            input_path = upload_dir / filename
            output_path = result_dir / filename

            # Save uploaded image
            with open(input_path, "wb+") as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            # Run complete detection and classification pipeline
            summary = analyse_image(input_path, output_path)

            report_dir = Path(settings.MEDIA_ROOT) / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)

            report_path = report_dir / f"{Path(filename).stem}_report.pdf"

            generate_pdf_report(
                input_path,
                output_path,
                report_path,
                summary
            )

            context["report_url"] = (
                settings.MEDIA_URL + "reports/" + report_path.name
            )

            context.update(summary)

            context["result_url"] = (
                settings.MEDIA_URL + "results/" + filename
            )

            context["original_url"] = (
                settings.MEDIA_URL + "uploads/" + filename
            )

            if summary["total"] == 0:
                context["error"] = (
                    "No solar panels were detected. "
                    "Try another image containing visible solar panels."
                )

        except Exception as e:
            context["error"] = f"Analysis failed: {str(e)}"

    return render(request, "detector/home.html", context)
