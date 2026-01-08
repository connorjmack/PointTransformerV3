# PROJECT CONTEXT: Deep Learning Integration for Coastal Cliff Monitoring
# DATE: 2026-01-02
# GOAL: Migrate from Random Forest (RF) to PointTransformerV3 (PTv3) for LiDAR Classification

## 1. THE STATUS QUO
We currently run a terrestrial/mobile LiDAR pipeline ("Cliff Processing Pipeline") monitoring erosion in Del Mar, CA.
- **Data Source:** High-density Mobile Laser Scanning (MLS).
- **File Format:** Massive .las files (5GB - 35GB per survey).
- **Current Stack:** Python, PDAL, Laspy, Scikit-Learn (Random Forest).
- **Current Classes:** Beach, Cliff, Vegetation.
- **Performance:** RF works well (>99% accuracy on simple geometry) but struggles with complex "flat" vegetation on cliff faces.

## 2. THE TARGET
We want to implement `PointTransformerV3` (a SOTA Transformer-based point cloud segmentation model) to replace or augment the Random Forest classifier.
- **Repo:** connorjmack/pointtransformerv3 (based on Pointcept).
- **Model Architecture:** Sparse Transformer with FlashAttention.

## 3. TECHNICAL CHALLENGES (The "Gaps")
1.  **The Semantic Gap:** PTv3 is pre-trained on autonomous driving datasets (NuScenes/Waymo) or indoor rooms (ScanNet). It does not know what "eroding sandstone" or "beach sand" is. It cannot be used out-of-the-box.
2.  **The Memory Gap:** Our files are 35GB. PTv3 (running on GPU) cannot process a whole file at once. It requires the data to be "tiled" into small 3D cubes (e.g., 5m x 5m blocks).

## 4. IMPLEMENTATION ROADMAP (Instructions for the LLM)

### PHASE A: Dataset Creation (Auto-Labeling)
*Goal: Create a formatted training dataset without manual labeling.*
1.  Take the existing .las files already classified by the Random Forest pipeline (where `classification` dimension is populated).
2.  Write a Python script (`01_prep_data.py`) that:
    - Reads the large .las file.
    - Slices it into 5m x 5m vertical columns or cubes.
    - Downsamples points to a fixed density (e.g., 0.02m grid).
    - Saves each block as a serialized file compatible with PTv3 (dict with `coord`, `color`, `segment`).

### PHASE B: Configuration & Training
*Goal: Fine-tune the model.*
1.  Define a Custom Dataset Class in the PTv3 codebase (inheriting from `DefaultDataset`).
2.  Create a config file (`config/cliff_segmentation.yaml`) mapping our 3 classes:
    - 0: Beach
    - 1: Cliff
    - 2: Vegetation
3.  Run training initializing with `NuScenes` pre-trained weights to speed up convergence.

### PHASE C: Inference Pipeline
*Goal: Run the model on new, raw data.*
1.  Write a wrapper script (`infer_cliff.py`) that:
    - Accepts a raw 35GB .las file.
    - Streams it, chopping it into temporary tiles.
    - Batches tiles to the GPU for PTv3 prediction.
    - Stitches the predicted labels back into the original .las file structure.

## 5. REQUEST TO ASSISTANT
Please help me write the code for **[INSERT PHASE HERE, e.g., PHASE A]**. I need Python code that uses `laspy` and `numpy` to handle the data formats described above.