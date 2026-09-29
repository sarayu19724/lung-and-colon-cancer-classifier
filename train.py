"""Two-stage EfficientNet training. Run python train.py --help."""
import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dataset import CLASSES, discover, save_manifest, split_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--groups-csv")
    parser.add_argument("--exploratory-image-split", action="store_true")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-epochs", type=int, default=8)
    parser.add_argument("--fine-tune-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.head_epochs, args.fine_tune_epochs) < 1:
        parser.error("Batch size and epoch counts must be positive.")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty. Choose a new --output for this run.")
    records = discover(args.data_dir, args.groups_csv, args.exploratory_image_split)
    splits = split_records(records, args.seed)
    output.mkdir(parents=True, exist_ok=True)
    save_manifest(splits, output / "split_manifest.csv")
    metadata = {
        "schema_version": 1, "classes": CLASSES, "image_size": 224,
        "preprocessing": "pillow_rgb_bilinear_0_255", "architecture": "EfficientNetB0",
        "created_at": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
        "evaluation_scope": "source-group split" if args.groups_csv else "exploratory image split",
        "split_counts": {name: dict(Counter(row["label"] for row in rows))
                         for name, rows in splits.items()},
        "duplicate_images": len(records) - len({row["sha256"] for row in records}),
        "training_complete": False,
    }
    def write_json(name, value):
        (output / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    write_json("metadata.json", metadata)
    print(json.dumps(metadata, indent=2))
    if args.audit_only:
        return

    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import classification_report, confusion_matrix
    from imaging import image_array

    keras.utils.set_random_seed(args.seed)
    root = Path(args.data_dir).resolve()

    def dataset(rows, training=False):
        paths = [str(root / row["path"]) for row in rows]
        labels = [CLASSES.index(row["label"]) for row in rows]
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            ds = ds.shuffle(len(rows), seed=args.seed, reshuffle_each_iteration=True)
        def decode(path, label):
            def load(raw_path):
                raw_path = raw_path.item() if hasattr(raw_path, "item") else raw_path
                return image_array(raw_path.decode("utf-8"), 224)
            image = tf.numpy_function(load, [path], tf.float32)
            image.set_shape((224, 224, 3))
            return image, label
        return ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = dataset(splits["train"], True)
    val_ds = dataset(splits["validation"])
    inputs = keras.Input((224, 224, 3), name="rgb_pixels")
    x = keras.layers.RandomFlip("horizontal_and_vertical", seed=args.seed)(inputs)
    x = keras.layers.RandomRotation(0.15, fill_mode="reflect", seed=args.seed + 1)(x)
    x = keras.layers.RandomZoom(0.1, seed=args.seed + 2)(x)
    base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet",
                                           input_shape=(224, 224, 3))
    base.trainable = False
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.35)(x)
    outputs = keras.layers.Dense(len(CLASSES), activation="softmax", dtype="float32")(x)
    model = keras.Model(inputs, outputs)

    def compile_model(rate):
        model.compile(optimizer=keras.optimizers.Adam(rate),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    def callbacks(stage):
        return [
            keras.callbacks.ModelCheckpoint(str(output / f"{stage}.keras"), monitor="val_loss",
                                            save_best_only=True),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
            keras.callbacks.CSVLogger(str(output / f"{stage}_history.csv")),
            keras.callbacks.TerminateOnNaN(),
        ]

    counts = Counter(row["label"] for row in splits["train"])
    weights = {i: len(splits["train"]) / (len(CLASSES) * counts[label])
               for i, label in enumerate(CLASSES)}
    compile_model(1e-3)
    model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs,
              callbacks=callbacks("head"), class_weight=weights)
    # Restore the actual checkpoint even when the stage simply exhausted its epochs.
    model.load_weights(output / "head.keras")
    head_loss = float(model.evaluate(val_ds, verbose=0)[0])
    base.trainable = True
    for layer in base.layers:
        layer.trainable = (layer.name.startswith(("block6", "block7", "top_"))
                           and not isinstance(layer, keras.layers.BatchNormalization))
    compile_model(1e-5)
    model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs,
              callbacks=callbacks("fine_tune"), class_weight=weights)
    model.load_weights(output / "fine_tune.keras")
    fine_loss = float(model.evaluate(val_ds, verbose=0)[0])
    selected = "fine_tune" if fine_loss < head_loss else "head"
    # Fine tuning is allowed to lose to the frozen-backbone model.
    model = keras.models.load_model(output / f"{selected}.keras", compile=False)
    model.save(output / "model.keras")
    test_ds = dataset(splits["test"])
    scores = model.predict(test_ds, verbose=1)
    y_true = np.array([CLASSES.index(row["label"]) for row in splits["test"]])
    y_pred = scores.argmax(axis=1)
    report = classification_report(y_true, y_pred, labels=list(range(len(CLASSES))),
                                   target_names=CLASSES, output_dict=True, zero_division=0)
    evaluation = {"evaluation_scope": metadata["evaluation_scope"],
                  "accuracy": float(np.mean(y_true == y_pred)),
                  "macro_f1": report["macro avg"]["f1-score"],
                  "classification_report": report,
                  "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
                  "test_samples": len(y_true), "selected_stage": selected,
                  "validation_loss": {"head": head_loss, "fine_tune": fine_loss}}
    write_json("evaluation.json", evaluation)
    metadata["training_complete"] = True
    metadata["selected_stage"] = selected
    write_json("metadata.json", metadata)
    print(f"Test accuracy: {evaluation['accuracy']:.4f}; macro F1: {evaluation['macro_f1']:.4f}")
    print(f"Saved model and evaluation to {output.resolve()}")


if __name__ == "__main__":
    main()
