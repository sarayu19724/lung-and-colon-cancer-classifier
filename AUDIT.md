# Local dataset and code audit

Audited on 28 September 2026. Dataset: `C:\GitHub\New folder\Lung-and-Colon-classifier\lung_colon_image_set`.

| Class | Images |
| --- | ---: |
| Colon adenocarcinoma | 5,000 |
| Benign colon tissue | 5,000 |
| Lung adenocarcinoma | 4,991 |
| Benign lung tissue | 5,000 |
| Lung squamous cell carcinoma | 5,000 |
| Total | 24,991 |

All images decoded successfully. There are **23,712 unique decoded RGB images** and **1,279 duplicate occurrences**. The hash includes image dimensions and EXIF-oriented RGB pixels, so matching pixels are detected even if file bytes differ. No identical-image conflicting labels were detected.

Reconstructing the supplied scripts' alphabetical per-class 80/20 `ImageDataGenerator` split reveals **370 distinct pixel hashes shared between training and validation**, affecting **373 of 4,998 validation images**. This is approximately 7.46% of the validation set. This reconstruction assumes the supplied scripts ran against the current local dataset; the saved models do not provide proof of their historical training data.

The replacement audit produced group-disjoint train/validation/test splits, grouping all exact duplicates together. No source-image or patient mapping was supplied, so this is an **exploratory image split**. Rotated/cropped augmented siblings can still cross splits. Exact duplicate grouping alone does not establish patient-independent generalization.

The original local folder layout already has five class directories directly under the dataset root, so its class discovery is correct. The original scripts have EarlyStopping and ReduceLROnPlateau, but no augmentation beyond rescaling. Their validation set also serves for model selection; there is no separate test set.

The original Flask app writes unvalidated upload filenames to disk, assumes the process working directory for model loading, enables debug mode when run directly, and displays only the winning class. The replacement uses in-memory validated image uploads, script-relative artifacts, a Waitress server, class-score visualization, and per-request latency.

Verification: seven automated tests passed, covering source-group splits, exact duplicate merging, conflicting labels, decoding/preprocessing, model-unavailable responses, malformed/oversized uploads, and successful prediction response formatting with a test double. Python compilation and JavaScript syntax checks passed. The Flask page was opened and visually checked. These checks do not demonstrate trained-model accuracy.

Full audit artifacts: `artifacts-audit/metadata.json`, `artifacts-audit/split_manifest.csv`, and `artifacts-audit/original_split_audit.json`. No new training or inference-quality result is claimed in this report.

Runtime limitation: TensorFlow installation stalled while downloading its 350.9 MB wheel and was stopped. Full training, checkpoint serialization, and real-model inference remain unverified. Flask dependencies were installed separately for the tested preview. Run the requirements installation successfully before training.
