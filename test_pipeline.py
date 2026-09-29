import csv
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from dataset import CLASSES, discover, split_records
from imaging import image_array, read_image


class PipelineTests(unittest.TestCase):
    def test_group_split_has_no_leakage_and_is_repeatable(self):
        records = [{"path": f"{label}/{group}/{variant}.png", "label": label,
                    "group": f"patient-{group}", "sha256": f"{label}-{group}-{variant}"}
                   for label in CLASSES for group in range(20) for variant in range(2)]
        splits = split_records(records)
        self.assertEqual(splits, split_records(records))
        groups = {key: {row["group"] for row in rows} for key, rows in splits.items()}
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["validation"] & groups["test"])
        self.assertEqual(sum(map(len, splits.values())), len(records))
        for rows in splits.values():
            self.assertEqual({row["label"] for row in rows}, set(CLASSES))

    def test_nested_classes_and_duplicate_group_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = []
            for i, label in enumerate(CLASSES):
                folder = root / ("colon_image_sets" if label.startswith("colon") else "lung_image_sets") / label
                folder.mkdir(parents=True)
                for j in range(2):
                    path = folder / f"{j}.png"
                    Image.new("RGB", (12, 12), (i * 40, 50, 100)).save(path)
                    mapping.append({"path": path.relative_to(root).as_posix(), "group": f"{label}-{j}"})
            group_file = root / "groups.csv"
            with group_file.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["path", "group"])
                writer.writeheader()
                writer.writerows(mapping)
            records = discover(root, group_file)
            self.assertEqual(len(records), 10)
            for label in CLASSES:
                self.assertEqual(len({row["group"] for row in records if row["label"] == label}), 1)
            with self.assertRaisesRegex(ValueError, "groups-csv"):
                discover(root)
            # An identical image in another class must not silently become training data.
            Image.new("RGB", (12, 12), (0, 50, 100)).save(root / "lung_image_sets" / "lung_n" / "0.png")
            with self.assertRaisesRegex(ValueError, "conflicting labels"):
                discover(root, exploratory=True)

    def test_image_contract_and_invalid_upload(self):
        buffer = BytesIO()
        Image.new("RGB", (24, 12), (255, 128, 0)).save(buffer, format="PNG")
        pixels = image_array(buffer.getvalue())
        self.assertEqual(pixels.shape, (224, 224, 3))
        self.assertEqual(pixels.dtype, np.float32)
        self.assertEqual(pixels[0, 0].tolist(), [255.0, 128.0, 0.0])
        with self.assertRaises(ValueError):
            read_image(b"not an image")
        with self.assertRaisesRegex(ValueError, "10 MB"):
            read_image(b"x" * (10 * 1024 * 1024 + 1))

    def test_insufficient_groups_rejected(self):
        records = [{"label": label, "group": "same-patient"} for label in CLASSES]
        with self.assertRaises(ValueError):
            split_records(records)


class FlaskTests(unittest.TestCase):
    def test_empty_model_and_upload_errors(self):
        from app import create_app
        with tempfile.TemporaryDirectory() as directory:
            client = create_app(directory).test_client()
            self.assertEqual(client.get("/").status_code, 200)
            self.assertFalse(client.get("/api/status").json["ready"])
            self.assertEqual(client.post("/api/predict").status_code, 503)

    def test_prediction_contract_and_latency(self):
        from app import create_app
        class FakeModel:
            def predict(self, pixels, verbose=0):
                assert pixels.shape == (1, 224, 224, 3)
                assert pixels.max() == 255
                return np.array([[0.05, 0.05, 0.8, 0.05, 0.05]])
        with tempfile.TemporaryDirectory() as directory:
            client = create_app(directory, bundle=(FakeModel(), {})).test_client()
            self.assertEqual(client.post("/api/predict").status_code, 400)
            self.assertEqual(client.post("/api/predict", data={"image": (BytesIO(b"bad"), "bad.png")}).status_code, 400)
            image = BytesIO()
            Image.new("RGB", (32, 32), (255, 0, 0)).save(image, format="PNG")
            image.seek(0)
            response = client.post("/api/predict", data={"image": (image, "../../image.png")})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json["predicted_class"], "lung_aca")
            self.assertEqual(len(response.json["scores"]), 5)
            self.assertGreaterEqual(response.json["processing_ms"], 0)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_oversize_upload(self):
        from app import create_app
        with tempfile.TemporaryDirectory() as directory:
            client = create_app(directory, bundle=(object(), {})).test_client()
            response = client.post("/api/predict", data=b"x" * (12 * 1024 * 1024),
                                   content_type="application/octet-stream")
            self.assertEqual(response.status_code, 413)


if __name__ == "__main__":
    unittest.main()
