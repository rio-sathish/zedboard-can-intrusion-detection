# Week 1 Plan — ZedBoard CAN Intrusion Detection

This week focuses on making the project environment ready and preparing the dataset workflow. No hardware synthesis is needed yet.

## Day 1 — Project setup
- Create the repository structure
- Add README and planning docs
- Prepare a Colab notebook for dataset work
- Confirm the target dataset files and project scope

## Day 2 — Dropbox-to-Colab workflow
- Collect Dropbox share links for the dataset files
- Convert each Dropbox link to a direct-download link using `?dl=1`
- Test file download in Colab with `wget` or Python `requests`
- Store files in a working `data/` folder

## Day 3 — Dataset inspection
- Load the CAN dataset files using pandas
- Print the first few rows of each file
- Check shape, columns, and dtypes
- Verify whether files are CSV or text-formatted

## Day 4 — Label and class balance checks
- Inspect the label/flag column
- Confirm the target classes:
  - Benign
  - DoS
  - Fuzzing
  - RPM-Spoof
- Check counts per class
- Note any imbalance between attack and normal samples

## Day 5 — Cleaning and parsing
- Handle missing values if present
- Check whether rows are fixed-width or variable-width
- If needed, prepare custom parsing logic for DLC-based message widths
- Normalize columns into a consistent structure

## Day 6 — Preprocessing preparation
- Build a first version of the sliding-window logic
- Convert 4 consecutive CAN messages into a 40-value feature vector
- Save cleaned outputs for reuse in later stages

## Day 7 — Review and checkpoint
- Confirm the dataset can be downloaded and read in Colab
- Confirm the inspection notebook runs end to end
- Save cleaned dataset artifacts to Google Drive or local storage
- Document any parsing issues or missing file format details

## Notes
- Week 1 is intentionally Windows and Colab friendly.
- Do not install Vivado, Vitis, or FINN yet for this stage.
- The later hardware flow will use Brevitas, ONNX, FINN, and ZedBoard PYNQ.

## Week 2 handoff (current)
- Use `notebooks/02_merge_balance_split_save.py` for merge/balance/split/save.
- Use `notebooks/03_training_starter.py` for CQMLP smoke training.
- Keep workflow Colab/Windows-first; no Vitis or synthesis yet.
