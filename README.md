# TissueLens — lung and colon histology classifier

An improved training pipeline and local Flask research app based on the supplied CNN scripts and the original local Flask app. The original project and its 24,991 images were found at `C:\GitHub\New folder\Lung-and-Colon-classifier`. Updated files live here in the writable workspace. No new model accuracy is claimed until training and evaluation complete.

## What changed

- Explicit detection of all five tissue classes, including the nested LC25000 layout. Your local dataset already has the correct flat layout. Loading the outer directory of the nested layout with `flow_from_directory` can incorrectly label the two organ folders instead of the five tissue classes.
- ImageNet-pretrained EfficientNetB0 at 224 pixels, training-only augmentation, global average pooling and dropout, followed by low-learning-rate fine tuning with frozen batch normalization. This is a candidate to evaluate, not a guarantee of the best model.
- Approximately 70/15/15 source-group splits, exact decoded-pixel duplicate auditing, fixed random seed, and a saved split manifest. Model selection uses validation loss only; test images are evaluated after selection.
- Best checkpoints from both stages are compared so fine tuning cannot replace a better first-stage model based on validation loss. Native `.keras` serialization, class order and preprocessing metadata accompany the model.
- Held-out accuracy, per-class precision/recall/F1, macro F1, confusion matrix, and training CSVs.
- Responsive Flask app with image preview, five-class scores, downloadable JSON, actual server processing time, evaluation metrics and clear missing-model states. Uploads are validated and processed in memory, replacing the original app's unvalidated filename-based disk writes. Waitress serves the app without Flask debug mode.

The pasted scripts also have no independent test set or source-group separation, and their `Flatten` layers create large dense heads. The 64-pixel/five-epoch version trades image detail and training time for speed. If the pasted blocks are literally joined as `print(...)import os`, that is a syntax error; keep separate scripts or replace them with this pipeline.

## Setup

Use Python 3.11 or 3.12. From this `improved` directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

TensorFlow CPU training may be slow. A supported GPU environment (for example a GPU notebook or WSL2 with the appropriate TensorFlow setup) is preferable for full training. The first training run downloads ImageNet weights. Internet access is required for dependencies and these weights. Start with batch size 16; reduce it to 8 if memory is limited.

## Dataset and grouping

Both layouts are supported:

```text
lung_colon_image_set/
  colon_image_sets/
    colon_aca/*.jpeg
    colon_n/*.jpeg
  lung_image_sets/
    lung_aca/*.jpeg
    lung_n/*.jpeg
    lung_scc/*.jpeg
```

Or put the five class folders directly under the dataset root. Use JPEG/PNG patches. Images without exactly one recognized class ancestor are rejected.

Create `groups.csv` from genuine provenance records, with paths relative to the dataset root:

```csv
path,group
colon_image_sets/colon_aca/example1.jpeg,patient_001
colon_image_sets/colon_aca/example1_augmented.jpeg,patient_001
```

Include **every** image. Use patient IDs if available; otherwise the original slide/source-image ID. All augmented descendants must share their source group. Use globally unique IDs. The code cannot infer real patient identity from filenames. Exact duplicate detection does not catch rotated, cropped, or other near-duplicate images. A source-image split is not automatically a patient-independent evaluation.

LC25000 contains augmented images, so a random image split can overstate generalization. If you cannot recover the original group mapping, explicitly use `--exploratory-image-split` for development and report that limitation; do not present its accuracy as new-patient performance.

Audit first (use a separate output directory for the audit):

```powershell
.\.venv\Scripts\python.exe train.py --data-dir "D:\datasets\lung_colon_image_set" --groups-csv groups.csv --audit-only --output artifacts-audit
```

Train:

```powershell
.\.venv\Scripts\python.exe train.py --data-dir "D:\datasets\lung_colon_image_set" --groups-csv groups.csv
```

Exploratory training when source groups are unavailable:

```powershell
.\.venv\Scripts\python.exe train.py --data-dir "D:\datasets\lung_colon_image_set" --exploratory-image-split
```

Choose a fresh `--output` directory for each run; existing runs are never overwritten. Inspect `split_manifest.csv` and class counts. Group sizes can cause split proportions to differ from 70/15/15. If there are too few independent groups to include all classes in each split, the script fails instead of silently using a misleading split.

## Run the app

```powershell
.\.venv\Scripts\python.exe app.py
```

Open `http://localhost:5000`. For a non-default training output, set `$env:MODEL_DIR = "D:\path\to\trained-artifacts"` first. The app needs the matching `model.keras`, `metadata.json`, and optional `evaluation.json`; it does not reuse old `.h5` models with unknown class order or preprocessing. It can launch without a trained model and preview uploads, but predictions remain disabled. Restart the server after training finishes to load the new model.

For this workspace's separately installed Flask preview dependencies, `.\run.ps1` starts the same app. The normal requirements installation remains necessary for model training. See `AUDIT.md` for measured duplicate leakage in the original split.

The Flask API exposes `GET /api/status` and `POST /api/predict` (multipart field `image`). It accepts images up to 10 MB / 20 megapixels. The response includes class scores and measured processing milliseconds. This is single-image patch inference; uploads of high-resolution images are resized to 224 pixels and are not analyzed as whole-slide scans.

## Résumé wording

Use these bullets after training and testing the finished pipeline:

- Trained TensorFlow CNN models to classify lung and colon histopathology images into five tissue classes: three malignant and two benign.
- Applied data augmentation, EarlyStopping, and adaptive learning-rate scheduling; evaluated performance using held-out accuracy, macro F1, and per-class recall.
- Built a Flask inference application with validated image uploads, consistent preprocessing, class-score visualization, and measured prediction latency.

Do not say “five malignant subtypes.” Keep “real-time” and numerical accuracy claims out until you have measured representative hardware latency and a defensible evaluation. The new work was added in September 2026; do not imply that these newly added features were implemented in May–July 2025 if they were not.

EfficientNet expects RGB pixel values in **0–255**, because rescaling is inside the model. Training and the app share the same Pillow EXIF orientation handling and bilinear resizing. Do not add `rescale=1./255` to this pipeline.

## Verification and limitations

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_pipeline.py
```

Tests cover nested class discovery, conflicting labels, duplicate/source-group separation, reproducible splits, and upload preprocessing/errors. Full training and model quality require your actual dataset and compute. There is no validated out-of-distribution detector or probability calibration. A high softmax score does not establish a diagnosis, and the model will assign one of its five classes even to unrelated images. This is a tissue-patch classifier, not a tumor-localization model or CT/X-ray classifier.

Compare this candidate against the original CNN on the **same independent splits**, using validation results to choose settings. Keep test results out of tuning decisions. Evaluate external patient/slide data before making real-world reliability claims.

References: [Keras EfficientNet fine-tuning guide](https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/) and [LC25000 dataset paper](https://arxiv.org/abs/1912.12142).
