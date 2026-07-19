# Week 2 Next Steps — Balance, Split, Train Starter

This stage formalizes the post-windowing workflow after Day 5 output (`X_dos`, `X_fuzzy`, `X_rpm` with 40-byte windows).

## 1) Merge windowed datasets
Use `combine_windowed_datasets` to build one multi-class dataset:
- Benign = `0`
- DoS = `1`
- Fuzzy = `2`
- RPM-Spoof = `3`

## 2) Balance by subsampling benign windows
Use `subsample_benign` with a fixed seed (`42` by default):
- keep all attack windows
- cap benign windows to `benign_multiplier * total_attack`
- default benign multiplier: `3`

## 3) Stratified split
Use `stratified_train_val_test_split`:
- train: `85%`
- validation: `10%`
- test: `5%`

## 4) Save processed dataset
Use `save_processed_dataset` to store:
- `X_train`, `y_train`
- `X_val`, `y_val`
- `X_test`, `y_test`

in a single `.npz` file for reuse.

## 5) Smoke-test training
Use `notebooks/03_training_starter.py`:
- loads `.npz`
- builds CQMLP skeleton (`src/model/cqmlp.py`)
- creates PyTorch DataLoaders
- runs a short smoke-test loop to validate pipeline wiring

## Notes
- Keep the workflow Colab/Windows friendly.
- Do not use Vitis in this stage.
- Hardware synthesis/deployment is intentionally deferred.
