# ZedBoard CAN Intrusion Detection

A professional final-year project repository for implementing a **ZedBoard-based CAN intrusion detection system** inspired by **Paper 5: Exploring Highly Quantised Neural Networks for Intrusion Detection in Automotive CAN**.

Current status:
- dataset loading and DLC-aware parsing completed
- 4-message sliding-window encoding completed (`40` features per window)
- merged multi-class windows prepared (`0=Benign, 1=DoS, 2=Fuzzy, 3=RPM-Spoof`)

The later pipeline will be:
**Dropbox/Colab dataset prep → Brevitas QAT → ONNX export → FINN build → ZedBoard PYNQ deployment**

## Target classes
- Benign
- DoS
- Fuzzing
- RPM-Spoof

> Note: Gear-spoof is intentionally skipped in this project scope.

## Platform choices
- **Runtime on ZedBoard:** PYNQ Linux
- **Hardware generation later:** FINN
- **Quantization/training later:** Brevitas
- **Development style for Week 1:** Windows + Colab friendly
- **Vitis:** not required for this project

## Current milestone (next Colab stage)
This stage formalizes:
1. merge the three windowed attack datasets
2. reproducible benign subsampling
3. stratified train/val/test split (`85/10/5`)
4. save final dataset (`.npz`) for training
5. run a CQMLP smoke-train starter

## Dataset notes
The CAN intrusion dataset files may require a little care when loading:
- Dropbox links should be converted to **direct-download links** for Colab
- Some files may have **variable-width rows** depending on DLC
- You may need custom parsing if `pandas.read_csv()` does not align columns correctly

## Run the next-step pipeline

### 1) Merge + balance + split + save
```bash
python notebooks/02_merge_balance_split_save.py \
  --dos data/dos_windows.npz \
  --fuzzy data/fuzzy_windows.npz \
  --rpm data/rpm_windows.npz \
  --output data/cqmlp_dataset.npz \
  --benign-multiplier 3 \
  --seed 42
```

Expected input format for each attack `.npz`: keys `X` and `y`.

### 2) Run training smoke test
```bash
python notebooks/03_training_starter.py \
  --dataset data/cqmlp_dataset.npz \
  --batch-size 1024 \
  --bit-width 2 \
  --device cpu
```

If Brevitas is available, quantized layers are used. If not, the starter falls back to standard PyTorch layers so the pipeline still runs in Colab/Windows.

## Repository structure
```text
zedboard-can-intrusion-detection/
├── README.md
├── requirements.txt
├── docs/
│   └── week1-plan.md
├── notebooks/
│   ├── 01_dataset_inspection_and_preprocessing.ipynb
│   ├── 02_merge_balance_split_save.py
│   └── 03_training_starter.py
└── src/
    ├── data/
    │   ├── __init__.py
    │   ├── can_loader.py
    │   └── processed_dataset.py
    └── model/
        ├── __init__.py
        └── cqmlp_starter.py
```

## Next phases
1. tune CQMLP training (loss/metrics/confusion matrix)
2. ONNX export
3. FINN compilation
4. ZedBoard deployment with PYNQ

## Goal
Produce a clean, realistic, and hardware-feasible CAN IDS implementation for ZedBoard.
