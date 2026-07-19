# Notebook/script workflow
#
# 01_dataset_inspection_and_preprocessing.ipynb
#   - Dropbox download
#   - DLC-aware parsing
#   - class balance checks
#   - cleaned data export
#
# 02_merge_balance_split_save.py
#   - merge windowed DoS/Fuzzy/RPM datasets
#   - reproducible benign subsampling
#   - stratified 85/10/5 split
#   - save split dataset to .npz
#
# 03_training_starter.py
#   - load saved .npz split dataset
#   - create DataLoaders
#   - build Paper 5 style CQMLP starter
#   - run one smoke-test epoch
