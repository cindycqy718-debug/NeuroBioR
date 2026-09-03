# Data Access Boundary

The Baidu Netdisk links and extraction codes are recorded in `DOWNLOADS.md`.

## Complete processed-data archive

The complete APAVA + ADFTD archive contains the processed per-subject arrays and labels for every subject assigned to the Medformer training, validation, and test splits.

Its directory contract is:

- `dataset/<DATASET>/Feature/feature_ID.npy`
- `dataset/<DATASET>/Label/label.npy`

APAVA contains 23 subjects and 5,967 windows of shape `[256,16]`.

ADFTD contains 88 subjects and 69,752 windows of shape `[256,19]`.

The archive includes source notes, split metadata, a file manifest, an example loader, and a full validation script.

## Fixed aligned TEST archive

The smaller aligned archive contains the exact TEST inputs expected by the public validation, re-scoring, and frozen-predictor replay commands.

Public model inputs do not expose labels or source subject identifiers through the Agent interface. Private scoring labels are used only by the local scorer.

| Dataset | TEST windows | Channels | Time points |
|---|---:|---:|---:|
| APAVA | 1,431 | 16 | 256 |
| ADFTD | 14,648 | 19 | 256 |

## TDBRAIN

TDBRAIN is access-controlled and is not redistributed.

Researchers authorized by the dataset owner may use the pinned preprocessing notebook at `third_party/Medformer/data_preprocessing/TDBRAIN_preprocessing.ipynb`.

The expected aligned TEST contract is 960 windows, 33 channels, 256 time points, 30 input batches, and 8 test subjects.

Do not upload raw or processed TDBRAIN signals to GitHub or a public file share unless the dataset owner explicitly permits redistribution.

## Rights and citation

The repository-level MIT license covers NeuroBioR-authored code and documentation. It does not automatically grant rights over third-party datasets.

Users must verify and follow each original provider's current access, citation, and redistribution terms.
