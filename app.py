from pathlib import Path
import re

import pandas as pd
from PIL import Image, ImageOps
from shiny import App, ui, render, reactive
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

IMAGE_SIZE = 64
MODEL_PATH = Path(__file__).parent / "model_0.13.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_LABELS = {0: "B1", 1: "B2"}


def default_threshold_from_model_name(model_path: Path) -> float:
    match = re.search(r"(\d*\.?\d+)", model_path.stem)
    if not match:
        return 0.5
    threshold = float(match.group(1))
    return min(max(threshold, 0.0), 1.0)


DEFAULT_THRESHOLD = default_threshold_from_model_name(MODEL_PATH)


def has_dark_background(image: Image.Image, threshold: float = 90.0) -> bool:
    gray = image.convert("L")
    arr = torch.tensor(list(gray.getdata()), dtype=torch.float32).reshape(gray.height, gray.width)

    # Use border pixels as a proxy for background brightness.
    top = arr[0, :]
    bottom = arr[-1, :]
    left = arr[:, 0]
    right = arr[:, -1]
    border_mean = torch.cat([top, bottom, left, right]).mean().item()
    return border_mean < threshold


def preprocess_image(image: Image.Image) -> torch.Tensor:
    image_rgb = image.convert("RGB")
    if has_dark_background(image_rgb):
        image_rgb = ImageOps.invert(image_rgb)
    return inference_transform(image_rgb)


def build_model() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


inference_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)

app_ui = ui.page_fluid(
    ui.h2("MMVD Radiograph Inference"),
    ui.p("Upload radiograph image(s) and run inference."),
    ui.input_file("images", "Radiograph images", multiple=True, accept=[".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]),
    ui.tags.details(
        ui.tags.summary("Advanced settings"),
        ui.input_slider("threshold", "Classification threshold", min=0.0, max=1.0, value=DEFAULT_THRESHOLD, step=0.01),
    ),
    ui.output_text_verbatim("status"),
    ui.download_button("download_predictions", "Download predictions (.csv)"),
    ui.output_table("predictions"),
)


def server(input, output, session):
    model = build_model() if MODEL_PATH.exists() else None

    @reactive.calc
    def run_inference():
        files = input.images()
        columns = ["file", "positive_probability", "threshold", "prediction"]
        threshold = float(input.threshold())
        if not files:
            return pd.DataFrame(columns=columns)
        if model is None:
            return pd.DataFrame(
                [{"file": "(error)", "positive_probability": None, "threshold": round(threshold, 2), "prediction": "Model checkpoint not found."}],
                columns=columns,
            )

        results = []
        for file_info in files:
            file_path = file_info["datapath"]
            file_name = file_info["name"]
            try:
                image = Image.open(file_path)
                tensor = preprocess_image(image).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    outputs = model(tensor)
                    probs = torch.softmax(outputs, dim=1)[0]
                    positive_prob = float(probs[1].item())
                    pred_idx = int(positive_prob >= threshold)

                results.append(
                    {
                        "file": file_name,
                        "positive_probability": round(positive_prob, 4),
                        "threshold": round(threshold, 2),
                        "prediction": CLASS_LABELS[pred_idx],
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "file": file_name,
                        "positive_probability": None,
                        "threshold": round(threshold, 2),
                        "prediction": f"Inference failed: {exc}",
                    }
                )

        return pd.DataFrame(results, columns=columns)

    @output
    @render.text
    def status():
        files = input.images()
        if not files:
            return "No images uploaded."
        if model is None:
            return "Model checkpoint missing."
        return f"Processed {len(files)} image(s)."

    @output
    @render.table
    def predictions():
        return run_inference()

    @render.download(filename="predictions.csv")
    def download_predictions():
        yield run_inference().to_csv(index=False)

app = App(app_ui, server)