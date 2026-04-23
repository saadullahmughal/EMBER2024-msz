# Original work by Robert J. Joyce, Gideon Miller, Phil Roth, Richard Zak,
# Elliott Zaresky-Williams, Hyrum Anderson, Edward Raff, and James Holt.
# Source: https://github.com/FutureComputing4AI/EMBER2024
# Reference: Joyce et al., "EMBER2024 - A Benchmark Dataset for Holistic
# Evaluation of Malware Classifiers", KDD 2025.
#
# Licensed under the Apache License, Version 2.0.
# http://www.apache.org/licenses/LICENSE-2.0
#
# Modified by M Saadullah Zafar, 2026
# Changes made to this file:
#   - vectorize_subset: accepts both file Paths and in-memory JSON line lists
#   - create_vectorized_features: added out_dir, filter_to_final_labels,
#     label_map_json, and subsets parameters
#   - Output paths now respect out_dir; directory is auto-created
#   - Subset vectorization is now selective (train/test/challenge)
#   - Label-map logic redesigned: supports loading from JSON, combines counts
#     across subsets, filters by class_min, assigns IDs via alphabetical sort,
#     and saves to label_map_<label_type>.json
#   - New filter_to_final_labels step drops samples lacking finalized labels
#   - Challenge vectorization now correctly passes label_type and label_map
#   - read_metadata bug fix: challenge DataFrame now built from challenge_records
#     instead of test_records

import os
import json
import multiprocessing
from pathlib import Path
from typing import Iterator

import lightgbm as lgb
import numpy as np
import polars as pl
import tqdm
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split

from .features import PEFeatureExtractor


ORDERED_COLUMNS = [
    "sha256",
    "tlsh",
    "first_submission_date",
    "last_analysis_date",
    "detection_ratio",
    "label",
    "file_type",
    "family",
    "family_confidence",
    "behavior",
    "file_property",
    "packer",
    "exploit",
    "group",
]


def raw_feature_iterator(file_paths: list[Path]) -> Iterator[str]:
    """
    Yield raw feature strings from the inputed file paths
    """
    for path in file_paths:
        with path.open("r") as fin:
            for line in fin:
                yield line


def gather_feature_paths(data_dir: Path | str, subset: str, filetype: str = None, week: str = None) -> list[Path]:
    """
    Gather paths to raw metadata .jsonl files in the given data_dir
    Supports filtering by train/test/challenge subset, file type, and/or data collection week
    """
    feature_paths = []
    for file_name in sorted(os.listdir(data_dir)):
        if not file_name.endswith(".jsonl"):
            continue
        if subset not in file_name:
            continue
        if filetype is not None and filetype not in file_name:
            continue
        if week is not None and week not in file_name:
            continue
        feature_paths.append(Path(os.path.join(data_dir, file_name)))

    if not len(feature_paths):
        raise ValueError("Did not find any .jsonl files matching criteria")
    return feature_paths


def read_label(raw_features_string: str, label_type: str) -> str:
    """
    Read the label or tag from raw features and return it
    """
    raw_features = json.loads(raw_features_string)
    label = raw_features[label_type]
    return label


def read_label_unpack(args):
    """
    Pass through function for unpacking read_label arguments
    """
    return read_label(*args)


def read_label_subset(raw_feature_paths: list[Path], nrows: int, label_type: str) -> set:
    """
    Read the unique labels/tags in the subset
    """
    # Distribute the vectorization work
    pool = multiprocessing.Pool()
    argument_iterator = (
        (raw_features_string, label_type)
        for _, raw_features_string in enumerate(raw_feature_iterator(raw_feature_paths))
    )
    label_counts = {}
    for labels in tqdm.tqdm(pool.imap_unordered(read_label_unpack, argument_iterator), total=nrows):
        if not isinstance(labels, list):
            labels = [labels]
        for label in labels:
            if label_counts.get(label) is None:
                label_counts[label] = 0
            label_counts[label] += 1
    return label_counts


def vectorize(irow: int, raw_features_string: str, X_path: str, y_path: str, extractor: PEFeatureExtractor, nrows: int, label_type: str = "label", label_map: dict = {}) -> None:
    """
    Vectorize a single sample of raw features and write to a large numpy file
    """
    raw_features = json.loads(raw_features_string)
    feature_vector = extractor.process_raw_features(raw_features)

    if label_type not in raw_features:
        raise ValueError("Invalid label_type!")
    label = raw_features[label_type]

    # Figure out what 'label' is
    if label is None and (label_type == "label" or label_type == "family"):
        y = np.memmap(y_path, dtype=np.int32, mode="r+", shape=nrows)
        y[irow] = -1
    elif isinstance(label, int): # Benign/Malicious labels (binary)
        y = np.memmap(y_path, dtype=np.int32, mode="r+", shape=nrows)
        y[irow] = label
    elif isinstance(label, str): # Family labels (multiclass)
        y = np.memmap(y_path, dtype=np.int32, mode="r+", shape=nrows)
        if label_map.get(label) is not None:
            y[irow] = label_map[label]
        else:
            y[irow] = -1
    elif isinstance(label, list): # Tags (multiclass, multilabel)
        y = np.memmap(y_path, dtype=np.int32, mode="r+", shape=(nrows, len(label_map.keys())))
        for l in label:
            if label_map.get(l) is not None:
                y[irow,label_map[l]] = 1
    else:
        raise ValueError("Unable to parse label format")

    X = np.memmap(X_path, dtype=np.float32, mode="r+", shape=(nrows, extractor.dim))
    X[irow] = feature_vector


def vectorize_unpack(args):
    """
    Pass through function for unpacking vectorize arguments
    """
    return vectorize(*args)


def vectorize_subset(
    X_path: Path,
    y_path: Path,
    raw_feature_paths: list,
    extractor: PEFeatureExtractor,
    nrows: int,
    label_type: str = "label",
    label_map: dict = {},
) -> None:
    """
    Vectorize a subset of data and write it to disk.

    `raw_feature_paths` may be a list of `Path` objects (files containing jsonl lines)
    or an in-memory list of raw json lines (strings). The function determines which
    and enumerates accordingly.
    """
    # Create space on disk to write features to
    X = np.memmap(X_path, dtype=np.float32, mode="w+", shape=(nrows, extractor.dim))
    if label_type == "label" or label_type == "family":
        y = np.memmap(y_path, dtype=np.float32, mode="w+", shape=nrows)
    else:
        y = np.memmap(y_path, dtype=np.float32, mode="w+", shape=(nrows, len(label_map.keys())))
    del X, y

    # Distribute the vectorization work
    pool = multiprocessing.Pool()
    # Accept either a list of file Paths or an in-memory list of raw feature strings
    if len(raw_feature_paths) and isinstance(raw_feature_paths[0], Path):
        iterator = enumerate(raw_feature_iterator(raw_feature_paths))
    else:
        iterator = enumerate(raw_feature_paths)

    argument_iterator = (
        (
            irow,
            raw_features_string,
            X_path,
            y_path,
            extractor,
            nrows,
            label_type,
            label_map,
        )
        for irow, raw_features_string in iterator
    )
    for _ in tqdm.tqdm(pool.imap_unordered(vectorize_unpack, argument_iterator), total=nrows):
        pass


def create_vectorized_features(
    data_dir: Path | str,
    out_dir: Path | str | None = None,
    label_type: str = "label",
    class_min: int = 10,
    filter_to_final_labels: bool = False,
    label_map_json: Path | str | None = None,
    subsets: list[str] | str | None = None,
) -> None:
    """
    Create feature vectors from raw features and write them to disk

    Arguments:
    data_dir - Path to the directory containing the dataset.
    out_dir - Optional path where vectorized outputs and label map are written.
              If None, outputs are written into `data_dir`.
    label_type - The type of classification problem.
    class_min - The minimum number of instances of a class in the dataset. Data
                points belonging to a class with fewer than class_min instances
                are ignored.
    filter_to_final_labels - If True and `label_type != "label"`, only samples
                that have at least one label present in the finalized `label_map`
                are retained for vectorization. Defaults to False (current behavior).
    label_map_json - Optional Path to a saved label map JSON file. If provided
                and `label_type != "label"`, this mapping will be used instead of
                deriving it from the dataset. The file must contain a JSON object
                mapping string labels to integer ids (e.g., {"ransomware": 0}).
    subsets - Optional subset or list of subsets to vectorize. Valid values are
                'train', 'test', and 'challenge'. If None (default), all three
                subsets will be vectorized.

    Valid label_types:
    label - malicious/benign (binary)
    family - malware family classification (multiclass)
    behavior - malware behavior prediction (multiclass, multi-label)
    file_property - malware file property prediction (multiclass, multi-label)
    packer - malware packer prediction (multiclass, multi-label)
    exploit - malware exploit prediction (multiclass, multi-label)
    group - malware threat group prediction (multiclass, multi-label)
    """
    # Ignore empty tags and self-describing file format tags
    ignore_tags = set(["", "win32", "win64", "elf", "linux", "pdf", "apk", "android"])

    extractor = PEFeatureExtractor()
    data_path: Path = Path(data_dir)
    out_path: Path = Path(out_dir) if out_dir is not None else data_path
    out_path.mkdir(parents=True, exist_ok=True)

    # Sanity check: label_map_json only applies to non-binary label types
    if label_map_json is not None and label_type == "label":

        raise ValueError(
            "label_map_json is only valid for non-binary label types (label_type != 'label')"
        )

    def sample_has_final_label(
        raw_features_string: str, label_type: str, label_map: dict
    ) -> bool:
        """Return True if the sample has at least one finalized label present in label_map."""
        raw = json.loads(raw_features_string)
        if label_type not in raw:
            return False
        label = raw[label_type]
        if label is None:
            return False
        if isinstance(label, int):
            return True
        if isinstance(label, str):
            return label in label_map
        if isinstance(label, list):
            return any(lbl in label_map for lbl in label)
        return False

    print("Preparing to vectorize raw features")

    # Normalize subsets parameter
    allowed_subsets = {"train", "test", "challenge"}
    if subsets is None:
        selected_subsets = ["train", "test", "challenge"]
    elif isinstance(subsets, str):
        selected_subsets = [subsets]
    else:
        selected_subsets = list(subsets)
    for s in selected_subsets:
        if s not in allowed_subsets:
            raise ValueError(
                f"Invalid subset '{s}' in subsets. Allowed: {allowed_subsets}"
            )

    if not selected_subsets:
        raise ValueError(
            "No subsets selected; choose from 'train', 'test', 'challenge'"
        )

    print(f"Vectorizing subsets: {selected_subsets}")

    # Gather paths and counts only for requested subsets
    feature_paths = {}
    nrows = {}
    if "train" in selected_subsets:
        feature_paths["train"] = gather_feature_paths(data_path, "train")
        nrows["train"] = sum([1 for fp in feature_paths["train"] for _ in fp.open()])
    if "test" in selected_subsets:
        feature_paths["test"] = gather_feature_paths(data_path, "test")
        nrows["test"] = sum([1 for fp in feature_paths["test"] for _ in fp.open()])
    if "challenge" in selected_subsets:
        feature_paths["challenge"] = gather_feature_paths(data_path, "challenge")
        nrows["challenge"] = sum(
            [1 for fp in feature_paths["challenge"] for _ in fp.open()]
        )

    # Map string labels/tags to numeric labels
    label_map = {}
    if label_type != "label": # No work needed for the default malicious/benign labels
        # If a saved label map JSON is provided, load and use it
        if label_map_json is not None:
            # Allow passing either a dict directly or a path to a JSON file
            if isinstance(label_map_json, dict):
                label_map = label_map_json
            else:
                label_map_path = Path(label_map_json)
                if not label_map_path.is_file():
                    raise ValueError(
                        f"Provided label_map_json does not exist: {label_map_path}"
                    )
                with label_map_path.open("r") as f:
                    label_map = json.load(f)
            if not isinstance(label_map, dict) or not all(
                isinstance(k, str) and isinstance(v, int) for k, v in label_map.items()
            ):
                raise ValueError(
                    "Loaded label_map_json must be a dict mapping string labels to integer ids"
                )
        else:
            # Read label counts for the selected subsets and build a combined mapping
            combined_counts = {}
            for s in selected_subsets:
                counts = read_label_subset(feature_paths[s], nrows[s], label_type)
                for lbl, cnt in counts.items():
                    combined_counts[lbl] = combined_counts.get(lbl, 0) + cnt

            # Collect labels that meet class_min and are not ignored
            included_labels = set()
            for label, count in combined_counts.items():
                if label in ignore_tags:
                    continue
                if count >= class_min:
                    included_labels.add(label)

            # Sort labels alphabetically before assigning numeric labels
            sorted_labels = sorted(included_labels)
            label_map = {lbl: idx for idx, lbl in enumerate(sorted_labels)}

            # Save label mapping as JSON
            label_map_path = out_path / f"label_map_{label_type}.json"
            with label_map_path.open("w") as f:
                json.dump(label_map, f, indent=2)

        # Optionally filter the selected datasets to only samples that contain at least one of the
        # finalized labels in `label_map`. This ensures only relevant samples are counted.
        if filter_to_final_labels:
            for s in list(feature_paths.keys()):
                filtered = [
                    line
                    for line in raw_feature_iterator(feature_paths[s])
                    if sample_has_final_label(line, label_type, label_map)
                ]
                feature_paths[s] = filtered
                nrows[s] = len(filtered)
                if nrows[s] == 0:
                    raise ValueError(
                        f"No {s} samples remain after filtering to finalized labels"
                    )

    # Vectorize only the requested subsets
    if "train" in selected_subsets:
        print("Vectorizing training set")
        X_train_path = out_path / "X_train.dat"
        y_train_path = data_path / "y_train.dat"
        vectorize_subset(
            X_train_path,
            y_train_path,
            feature_paths["train"],
            extractor,
            nrows["train"],
            label_type,
            label_map,
        )

    if "test" in selected_subsets:
        print("Vectorizing test set")
        X_test_path = out_path / "X_test.dat"
        y_test_path = out_path / "y_test.dat"
        vectorize_subset(
            X_test_path,
            y_test_path,
            feature_paths["test"],
            extractor,
            nrows["test"],
            label_type,
            label_map,
        )

    if "challenge" in selected_subsets:
        print("Vectorizing challenge set")
        X_challenge_path = out_path / "X_challenge.dat"
        y_challenge_path = out_path / "y_challenge.dat"
        vectorize_subset(
            X_challenge_path,
            y_challenge_path,
            feature_paths["challenge"],
            extractor,
            nrows["challenge"],
            label_type,
            label_map,
        )


def read_vectorized_features(data_dir: Path | str, subset: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """
    Read vectorized features into memory mapped numpy arrays
    """
    data_path: Path = Path(data_dir)
    X_path = data_path / f"X_{subset}.dat"
    y_path = data_path / f"y_{subset}.dat"

    if not os.path.isfile(X_path):
        raise ValueError(f"Invalid subset file: {X_path}")
    if not os.path.isfile(y_path):
        raise ValueError(f"Invalid subset file: {y_path}")

    extractor = PEFeatureExtractor()
    ndim: int = extractor.dim
    X = np.memmap(X_path, dtype=np.float32, mode="r")
    X = np.array(X).reshape(-1, ndim)
    N: int = X.shape[0]
    y = np.memmap(y_path, dtype=np.int32, mode="r")
    y = np.array(y)
    if y.shape[0] > N:
        y = y.reshape(N, -1)

    return X, y


def read_metadata_record(raw_features_string: str) -> dict:
    """
    Decode a raw features string and return the metadata fields
    """
    all_data = json.loads(raw_features_string)
    metadata_keys = set(ORDERED_COLUMNS)
    return {k: all_data[k] for k in all_data.keys() & metadata_keys}


def read_metadata(data_dir: Path | str) -> pl.DataFrame:
    """
    Write metadata to a csv file and return its dataframe
    """
    pool = multiprocessing.Pool()
    data_path: Path = Path(data_dir)

    train_feature_paths = gather_feature_paths(data_path, "train")
    train_records = list(pool.imap(read_metadata_record, raw_feature_iterator(train_feature_paths)))
    train_metadf = pl.DataFrame(train_records).with_columns(subset=pl.lit("train")).select(ORDERED_COLUMNS)

    test_feature_paths = gather_feature_paths(data_path, "test")
    test_records = list(pool.imap(read_metadata_record, raw_feature_iterator(test_feature_paths)))
    test_metadf = pl.DataFrame(test_records).with_columns(subset=pl.lit("test")).select(ORDERED_COLUMNS)

    challenge_feature_paths = gather_feature_paths(data_path, "challenge")
    challenge_records = list(pool.imap(read_metadata_record, raw_feature_iterator(challenge_feature_paths)))
    challenge_metadf = (
        pl.DataFrame(challenge_records)
        .with_columns(subset=pl.lit("challenge"))
        .select(ORDERED_COLUMNS)
    )

    return train_metadf, test_metadf, challenge_metadf


def optimize_model(data_dir: Path | str) -> dict:
    """
    Run a grid search to find the best LightGBM parameters
    """
    # Read data
    X_train, y_train = read_vectorized_features(data_dir, "train")
    train_rows = y_train != -1
    X_train_labeled = X_train[train_rows]
    y_train_labeled = y_train[train_rows]

    # Score by ROC AUC
    # We're interested in low FPR rates, so we'll consider only the AUC for FPRs in [0,5e-3]
    score = make_scorer(roc_auc_score, max_fpr=5e-3)

    # Each row in X_train appears in chronological order of "first_seen_date" so this works for
    # progrssive time series splitting
    progressive_cv = TimeSeriesSplit(n_splits=3).split(X_train_labeled)

    fit_params = {"categorical_feature": [2, 3, 4, 5, 6, 701, 702]}
    param_grid = {
        "boosting_type": ["gbdt"],
        "objective": ["binary"],
        "num_iterations": [500, 1000],
        "learning_rate": [0.005, 0.05],
        "num_leaves": [512, 1024, 2048],
        "feature_fraction": [0.5, 0.8, 1.0],
        "bagging_fraction": [0.5, 0.8, 1.0],
    }
    grid = GridSearchCV(
        estimator=lgb.LGBMClassifier(n_jobs=-1, verbose=-1),
        cv=progressive_cv,
        param_grid=param_grid,
        scoring=score,
        n_jobs=1,
        verbose=3,
    )
    grid.fit(X_train_labeled, y_train_labeled, **fit_params)

    return grid.best_params_


def train_model(data_dir: Path | str, params: dict = {}) -> lgb.Booster:
    """
    Train LightGBM model on the vectorized features.
    """
    # Read data
    X, y = read_vectorized_features(data_dir, "train")

    # Verify that y_train is not formatted for multi-label classification
    if len(y.shape) != 1:
        raise ValueError("Encounted y_train with invalid shape. Use train_ovr_model() instead.")

    # Ignore files without a label/tag
    num_classes = np.max(y) + 1
    X = X[y != -1, :]
    y = y[y != -1]

    # Use a stratified split to make a validation set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y)
    train_set = lgb.Dataset(X_train, y_train, categorical_feature=[2, 3, 4, 5, 6, 701, 702])
    val_set = lgb.Dataset(X_val, y_val, reference=train_set, categorical_feature=[2, 3, 4, 5, 6, 701, 702])

    # Binary classification
    if num_classes == 2:
        return lgb.train(params, train_set, valid_sets=val_set)

    # Multiclass classification
    lgbm_params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss"
    }
    params.update(lgbm_params)
    return lgb.train(params, train_set, valid_sets=val_set)


def train_ovr_model(data_dir: Path | str, params: dict = {}) -> lgb.Booster:
    """
    Returns a list of One-vs-Rest (OvR) LightGBM classifiers trained on the vectorized features.
    """
    # Read data
    X, y = read_vectorized_features(data_dir, "train")

    # Verify that y_train is not formatted for multi-label classification
    if len(y.shape) != 2:
        raise ValueError("Encounted y_train with invalid shape. Use train_model() instead.")

    # OvR Multilabel classification
    lgbm_models = []
    for i in range(y.shape[1]):
        lgbm_params = {
            "objective": "binary",
            "is_unbalance": True,
        }
        params.update(lgbm_params)
        y_i = y[:, i]
        X_train, X_val, y_train, y_val = train_test_split(X, y_i, test_size=0.1, stratify=y_i)
        train_set = lgb.Dataset(X_train, y_train, categorical_feature=[2, 3, 4, 5, 6, 701, 702])
        val_set = lgb.Dataset(X_val, y_val, reference=train_set, categorical_feature=[2, 3, 4, 5, 6, 701, 702])
        lgbm_models.append(lgb.train(params, train_set, valid_sets=val_set))
    return lgbm_models


def predict_sample(lgbm_model: lgb.Booster, file_data: bytes) -> float:
    """
    Predict a PE file with an LightGBM model
    """
    extractor = PEFeatureExtractor()
    features = np.array(extractor.feature_vector(file_data), dtype=np.float32)
    predict_result: np.ndarray = lgbm_model.predict([features])
    return float(predict_result[0])
