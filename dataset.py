"""Discover five tissue classes, audit duplicates, and split whole source groups."""
import csv
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

from imaging import read_image

CLASSES = ["colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"]
DISPLAY_NAMES = {
    "colon_aca": "Colon adenocarcinoma",
    "colon_n": "Benign colon tissue",
    "lung_aca": "Lung adenocarcinoma",
    "lung_n": "Benign lung tissue",
    "lung_scc": "Lung squamous cell carcinoma",
}


def discover(root, groups_csv=None, exploratory=False):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory not found: {root}")
    if not groups_csv and not exploratory:
        raise ValueError("Provide --groups-csv with path,group columns, or explicitly use "
                         "--exploratory-image-split. Augmented images need source/patient groups.")
    groups = {}
    if groups_csv:
        with open(groups_csv, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not {"path", "group"}.issubset(reader.fieldnames or []):
                raise ValueError("Group CSV must have path and group columns.")
            for row in reader:
                path, group = row["path"].strip().replace("\\", "/"), row["group"].strip()
                if not path or not group or path in groups:
                    raise ValueError("Group CSV contains a blank value or duplicate path.")
                groups[path] = group
    records, seen = [], {}
    parents = {}

    def find(group):
        parents.setdefault(group, group)
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        relative = path.relative_to(root).as_posix()
        labels = set(path.relative_to(root).parts[:-1]) & set(CLASSES)
        if len(labels) != 1:
            raise ValueError(f"Image must be inside exactly one of the five class folders: {relative}")
        label = labels.pop()
        if groups_csv and relative not in groups:
            raise ValueError(f"Missing source group for {relative}")
        image = read_image(path)
        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        group = groups[relative] if groups_csv else digest
        find(group)
        if digest in seen:
            previous = seen[digest]
            if previous["label"] != label:
                raise ValueError(f"Identical images have conflicting labels: {relative}")
            # Even incorrectly distinct source IDs cannot put identical pixels across splits.
            parents[find(group)] = find(previous["group"])
        record = {"path": relative, "label": label, "group": group, "sha256": digest}
        records.append(record)
        seen[digest] = record
        if len(records) % 1000 == 0:
            print(f"Audited {len(records):,} images...", flush=True)
    counts = Counter(row["label"] for row in records)
    if set(counts) != set(CLASSES):
        raise ValueError(f"Expected all five tissue classes. Found: {dict(counts)}")
    if groups_csv and set(groups) != {row["path"] for row in records}:
        raise ValueError("Group CSV has paths not found in the dataset; check relative paths.")
    for row in records:
        row["group"] = find(row["group"])
    return records


def split_records(records, seed=42):
    """Select an approximately 70/15/15 group split using labels only, never model scores."""
    grouped = defaultdict(list)
    for row in records:
        grouped[row["group"]].append(row)
    keys = sorted(grouped)
    if len(keys) < 3:
        raise ValueError("At least three independent groups are needed.")
    total = Counter(row["label"] for row in records)
    rng, best, best_cost = random.Random(seed), None, float("inf")
    n_test = max(1, round(len(keys) * 0.15))
    n_val = max(1, round(len(keys) * 0.15))
    for _ in range(256):
        rng.shuffle(keys)
        partitions = [keys[n_test + n_val:], keys[n_test:n_test + n_val], keys[:n_test]]
        splits = [[row for key in part for row in grouped[key]] for part in partitions]
        counts = [Counter(row["label"] for row in part) for part in splits]
        if any(set(count) != set(CLASSES) for count in counts):
            continue
        cost = sum(abs(count[label] / total[label] - ratio)
                   for count, ratio in zip(counts, [0.7, 0.15, 0.15]) for label in CLASSES)
        if cost < best_cost:
            best, best_cost = splits, cost
    if best is None:
        raise ValueError("Cannot create three group-disjoint splits containing every class. "
                         "Add more independent groups per class.")
    return dict(zip(["train", "validation", "test"], best))


def save_manifest(splits, destination):
    with open(destination, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "group", "sha256", "split"])
        writer.writeheader()
        for split, rows in splits.items():
            writer.writerows(dict(row, split=split) for row in rows)
