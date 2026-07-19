# ZedBoard CAN Intrusion Detection

A professional final-year project repository for implementing a **ZedBoard-based CAN intrusion detection system** inspired by **Paper 5: Exploring Highly Quantised Neural Networks for Intrusion Detection in Automotive CAN**.

This repository is currently scoped for **Week 1 only**:
- dataset access and inspection
- Dropbox-to-Colab workflow
- pandas-based cleaning and class balance checks
- preprocessing preparation for later model training

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

## What this repo contains right now
- Week 1 project plan
- Dataset inspection and preprocessing starter notebook
- Lightweight Python helper structure for CAN data processing
- Minimal dependencies for data work

## Dataset notes
The CAN intrusion dataset files may require a little care when loading:
- Dropbox links should be converted to **direct-download links** for Colab
- Some files may have **variable-width rows** depending on DLC
- You may need custom parsing if `pandas.read_csv()` does not align columns correctly

## Week 1 deliverable
By the end of Week 1, you should have:
- data loaded from Dropbox into Colab
- raw files inspected
- class counts checked
- cleaned data saved for reuse
- a preprocessing pipeline ready for model training

## Repository structure
```text
zedboard-can-intrusion-detection/
├── README.md
├── requirements.txt
├── docs/
│   └── week1-plan.md
├── notebooks/
│   └── 01_dataset_inspection_and_preprocessing.ipynb
└── src/
    └── data/
        ├── __init__.py
        └── can_loader.py
```

## Next phases
When Week 1 is complete, the project will move to:
1. data preprocessing finalization
2. Brevitas quantization-aware training
3. ONNX export
4. FINN compilation
5. ZedBoard deployment with PYNQ

## Goal
Produce a clean, realistic, and hardware-feasible CAN IDS implementation for ZedBoard.
