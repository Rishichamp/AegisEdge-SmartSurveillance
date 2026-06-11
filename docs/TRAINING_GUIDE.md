# Anomaly Classifier Model Training Guide
=========================================

This document provides a comprehensive guide on preparing custom datasets, training the LSTM Action Classifier model, and tuning hyperparameters for custom real-world surveillance deployment.

---

## 1. Custom Dataset Collection Pipeline

To train the LSTM classifier for a custom environment, you must convert raw video recordings (e.g. fight clips, normal walking, shoplifting events) into sequence feature files.

### Step 1.1. Video Recording Split
* Record or gather video clips of each behavior category:
  * `0 (normal)`
  * `1 (fight)`
  * `2 (theft)`
  * `3 (intrusion)`
* Keep clips short (typically 2 to 10 seconds each) and ensure they are recorded at a standard frame rate (e.g., 30 FPS).

### Step 1.2. Preprocessing & Feature Extraction
We must extract the person regions, run them through MobileNetV3 to extract posture features, and group them into sequences of length 16.

Write a custom script (e.g., `scripts/setup_dataset.py`) to process raw videos:
1. Use `ObjectDetector` to detect and track persons in each clip.
2. For each active `track_id`, crop the person from the frame.
3. Pass the crop through `ROIFeatureExtractor.extract_features()` to get a 576-dimensional vector.
4. Maintain a sliding buffer of 16 frames. Once the buffer is full, save the $16 \times 576$ array as a feature sequence.
5. Export all extracted sequences to NumPy arrays:
   - `sequences.npy` of shape `(num_sequences, 16, 576)`
   - `labels.npy` of shape `(num_sequences,)` with class indices (0 to 3).

---

## 2. Running the Training Script

Once your NumPy files are preprocessed and saved under a data directory (e.g. `data/preprocessed/`), launch the training loop:

```bash
python scripts/train_anomaly_model.py \
    --data_dir data/preprocessed/ \
    --epochs 30 \
    --batch_size 64 \
    --lr 0.0005 \
    --hidden_size 256 \
    --dropout 0.4 \
    --output_path models/custom_lstm_model.pth
```

### Explanations of Training Arguments:
* `--lr`: Learning rate (typical default is `0.001` or `0.0005`). Lower values converge slower but reduce oscillation around the local minima.
* `--hidden_size`: The size of the LSTM hidden state. Increase to `512` for highly complex action profiles; decrease to `128` for lighter compute profiles.
* `--dropout`: Probability of randomly dropping neural connections. Higher values (e.g. `0.4` or `0.5`) reduce overfitting on small datasets.
* `--batch_size`: Number of sequences processed in a single optimizer step. Keep at `32` or `64`.

---

## 3. Resolving Performance Issues (Tuning Checklist)

When analyzing your model validation output logs, check for these patterns:

### Case A: High Training Loss & High Validation Loss (Underfitting)
* **Diagnosis:** The model is too simple to learn the data patterns.
* **Fixes:**
  * Increase model capacity: Set `--hidden_size 512` or add a third LSTM layer (`--num_layers 3`).
  * Increase epochs: Train for `50` or `100` epochs instead of `10`.
  * Raise learning rate: Set `--lr 0.002` (but monitor for divergence).

### Case B: Low Training Loss & High Validation Loss (Overfitting)
* **Diagnosis:** The model has "memorized" the training set and fails to generalize to unseen test videos.
* **Fixes:**
  * Increase regularization: Increase dropout (`--dropout 0.5`).
  * Add Weight Decay: In your optimizer code, ensure L2 regularization (`weight_decay=1e-4`) is active.
  * Data Augmentation: Add noise or apply spatial jitters (crops, rotations) to the raw person crops before feature extraction.
  * Early Stopping: Terminate training once the validation loss starts rising, even if training loss continues to drop.
