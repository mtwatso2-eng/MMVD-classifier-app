# MMVD Classifier App

This project is a Python Shiny web app for radiograph inference using a trained ResNet18 binary classifier.

## What the App Does

- Accepts one or more radiograph images (`png`, `jpg`, `jpeg`, `bmp`, `tif`, `tiff`)
- Runs model inference and outputs:
  - file name
  - positive-class probability
  - threshold used
  - predicted class label (`B1` or `B2`)
- Includes an **Advanced settings** section with a classification threshold slider (`0.0` to `1.0`)
- Defaults the threshold to the numeric suffix in the model filename (e.g. `model_0.65.pth` -> `0.65`)
- Applies simple conditional preprocessing:
  - detects dark backgrounds using border intensity
  - inverts image only when background appears dark
- Allows downloading the results table as a CSV file

## Requirements

- Python 3.10+ recommended
- A model checkpoint file named `model_0.65.pth` in the project root (same folder as `app.py`)

## Run Locally

1. (Optional) Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
python -m shiny run --reload app.py
```

4. Open the local URL shown in terminal (typically `http://127.0.0.1:8000`).

## Notes

- If the model file is missing, the app will show a checkpoint missing message.
- CSV export contains the same columns shown in the results table.