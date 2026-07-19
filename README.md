# ZedBoard CAN Intrusion Detection

A final-year project repository for implementing a **ZedBoard-based CAN intrusion detection system** inspired by **Paper 5: Exploring Highly Quantised Neural Networks for Intrusion Detection in Automotive CAN**.

## Current milestone
Week 1 data loading and parsing are complete. The repository now includes reproducible scaffolding for the next stage:
- merge the three windowed attack datasets into a combined multi-class dataset
- subsample benign windows with a fixed RNG seed
- create reproducible 85/10/5 stratified train/val/test splits
- save and reload split datasets from `.npz`
- run a starter CQMLP smoke-test training loop

## Target classes
- `0`: Benign
- `1`: DoS
- `2`: Fuzzy
- `3`: RPM-Spoof

> Note: Gear-spoof is intentionally skipped in this project scope.

## Platform choices
- **Runtime on ZedBoard:** PYNQ Linux
- **Training/quantization framework:** Brevitas (with fallback placeholders in local starter code)
- **Development style:** Windows + Colab friendly
- **Vitis:** not required at this stage

## Repository structure
```text
zedboard-can-intrusion-detection/
├── README.md
├── requirements.txt
├── docs/
│   ├── week1-plan.md
│   └── week2-next-steps.md
├── notebooks/
│   ├── README.md
│   ├── 02_balance_split_export.py
│   └── 03_training_starter.py
└── src/
    ├── data/
    │   ├── __init__.py
    │   ├── can_loader.py
    │   └── processed_dataset.py
    └── model/
        ├── __init__.py
        └── cqmlp.py
```

## Quick start for next stage
1. Generate windowed arrays in Colab (`X_dos/y_dos`, `X_fuzzy/y_fuzzy`, `X_rpm/y_rpm`) and save as `.npy` files.
2. Run:
   ```bash
   python notebooks/02_balance_split_export.py \
     --dos-x /path/X_dos.npy --dos-y /path/y_dos.npy \
     --fuzzy-x /path/X_fuzzy.npy --fuzzy-y /path/y_fuzzy.npy \
     --rpm-x /path/X_rpm.npy --rpm-y /path/y_rpm.npy \
     --output /path/cqmlp_dataset.npz
   ```
3. Run smoke-test training:
   ```bash
   python notebooks/03_training_starter.py --dataset /path/cqmlp_dataset.npz
   ```

## Next phases
1. Brevitas quantization-aware training refinement
2. ONNX export
3. FINN compilation
4. ZedBoard deployment with PYNQ

## Goal
Produce a clean, realistic, hardware-feasible CAN IDS implementation for ZedBoard.
