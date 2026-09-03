# Companion Data Downloads

Two companion archives are provided. They support different reproduction paths and should not be confused.

## 1. Complete processed APAVA + ADFTD data

- Recommended filename: `NeuroBioR_DATA.zip`
- Baidu Netdisk: https://pan.baidu.com/s/1ZJgXS6-yDihik-n_VRKL0A?pwd=rufk
- Extraction code: `rufk`

This archive contains all processed subject arrays and labels assigned to the Medformer training, validation, and test splits for APAVA and ADFTD.

After extraction, read `README_DATA_EN.md` or `README_DATA_CN.md`, then run:

- `python -m pip install -r requirements_data.txt`
- `python validate_data.py --full-scan`

Use this archive for the complete Medformer-aligned dataset layout, baseline training, split inspection, and data-loading verification.

## 2. Fixed aligned TEST data

- Recommended filename: `NeuroBioR_Aligned_TEST_Data.zip`
- Baidu Netdisk: https://pan.baidu.com/s/11-viJSXdNbrEmVNFEgFN-Q?pwd=ewa4
- Extraction code: `ewa4`

This archive contains the exact aligned APAVA and ADFTD TEST tensors expected by the repository's validation, re-scoring, and frozen-predictor replay commands.

Extract it beside the repository. The expected directories are:

- `work/NeuroBioR/`
- `work/NeuroBioR_Aligned_TEST_Data_v1.1/data/`

## Access boundary

TDBRAIN is not included in either public archive. Authorized users must obtain it from its original provider.

Dataset use remains governed by the original dataset providers. See `docs/DATA_ACCESS.md` and `LICENSES_AND_DATA_TERMS.md`.
