# Changes in this Fork

Maintained by M Saadullah Zafar (2026).
Fork of [EMBER2024](https://github.com/FutureComputing4AI/EMBER2024) by Joyce et al., KDD 2025.

---

## src/features.py

- Updated Authenticode/signature parsing code for signify 0.9.2 API compatibility.
  The signify package introduced breaking changes in its public API between 0.7.x
  and 0.9.x; this file has been updated accordingly.

## src/model.py

- `vectorize_subset`: now accepts both file `Path` objects and in-memory JSON line
  lists, with dynamic iterator selection. Useful for low-memory batch workflows.
- `create_vectorized_features`: added four new parameters:
  - `out_dir` — write vectorized outputs to a separate directory (auto-created)
  - `subsets` — selectively vectorize only `train`, `test`, and/or `challenge`
  - `label_map_json` — load a pre-saved label map instead of deriving from data
  - `filter_to_final_labels` — drop samples with no label in the finalized map
- Label-map logic redesigned:
  - Supports loading from a JSON file (with validation)
  - Combines label counts across all selected subsets before filtering
  - Assigns numeric IDs via alphabetical sort for deterministic reproducibility
  - Saves derived map to `label_map_<label_type>.json`
- Challenge vectorization now correctly passes `label_type` and `label_map`
  through to `vectorize_subset` (previously used defaults silently)
- **Bug fix:** `read_metadata` challenge DataFrame was being built from
  `test_records` instead of `challenge_records`

## pyproject.toml

- All runtime dependencies pinned to exact tested versions (see below)
- Removed duplicate `lightgbm` entry
- Python requirement tightened to `>=3.12.10,<3.13`

| Package | Was | Now |
|---|---|---|
| lightgbm | >=4.5.0 (listed twice) | ==4.6.0 |
| pefile | >=2024.8.26 | ==2024.8.26 |
| polars | >=1.8.2 | ==1.37.1 |
| scikit-learn | >=1.5.1 | ==1.8.0 |
| tqdm | >=4.66.5 | ==4.67.2 |
| signify | >=0.7.1 | ==0.9.2 |
| huggingface_hub | >=0.32.4 | ==1.3.7 |
| matplotlib | >=3.10.3 | ==3.10.8 |

## setup.cfg

- Synced all runtime dependencies to match `pyproject.toml` (exact pinned versions)
- Corrected `name` from `EMBER2024` to `thrember` (was mismatched with `pyproject.toml`)
- Corrected `version` from `1.0.0` to `0.1.0`
- Removed `setuptools` and `setuptools-scm` from `install_requires` (build-time only;
  belong in `[build-system]`, not as runtime dependencies)
- Tightened `python_requires` to `>=3.12.10,<3.13`