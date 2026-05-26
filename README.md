# MDOC

Minimal public implementation of Medical Deep One-Class Classification.

## Files

`model.py` contains the dual-branch MDOC network.

`data.py` contains a generic metadata-based image loader.

`train.py` trains and evaluates the main MDOC experiment.

## Data Format

Prepare a CSV file with at least three columns:

```csv
path,label,split
images/sample_001.png,normal,train
images/sample_002.png,normal,test
images/sample_003.png,abnormal,test
```

The training split should contain the target class. The test split may contain both target-class and out-of-class samples. Image paths can be absolute or relative to `--data-root`.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python train.py \
  --data-root /path/to/images \
  --metadata /path/to/metadata.csv \
  --target-label normal \
  --output-dir runs/normal \
  --channels 1
```

Use `--channels 3` for RGB images.

## Outputs

`best.pt` stores the best checkpoint by AUC.

`history.csv` stores epoch-wise training and evaluation metrics.

`scores.csv` stores test labels and reconstruction-error anomaly scores.

`metrics.json` stores the final epoch metrics.
