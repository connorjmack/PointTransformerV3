# GEMINI.md - Project Context: Point Transformer V3 for Coastal Cliff Monitoring

## Project Overview
This project adapts **Point Transformer V3 (PTv3)**, a state-of-the-art sparse transformer architecture, for the classification of high-density Mobile Laser Scanning (MLS) LiDAR data of coastal cliffs (specifically Del Mar, CA). The goal is to migrate from a Random Forest-based pipeline to a deep learning approach to better handle complex vegetation on cliff faces.

### Main Technologies
- **Model:** Point Transformer V3 (PTv3)
- **Frameworks:** PyTorch, SpConv (Sparse Convolution), FlashAttention
- **Data Handling:** Laspy, PDAL, NumPy
- **Key Architecture:** Sparse Transformer with Z-order/Hilbert serialization for efficient attention.

## Building and Running

### Environment Setup
1. **Conda Environment:**
   ```bash
   conda create -n pointcept python=3.8 -y
   conda activate pointcept
   conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
   pip install spconv-cu118
   pip install torch-geometric
   ```
2. **Flash Attention (Optional but Recommended):**
   Refer to the [FlashAttention repo](https://github.com/Dao-AILab/flash-attention) for installation.

### Key Commands
- **Training:** (TODO: Implement `scripts/train.sh` or equivalent for cliff dataset)
- **Data Prep:** (TODO: Create `01_prep_data.py` for tiling and voxelization)
- **Inference:** (TODO: Create `infer_cliff.py` for tiled inference on large .las files)

## Development Conventions & Data Pipeline

### 1. Data Structure (`data_dict`)
The model expects a dictionary (or `Point` object) with the following:
- `coord`: (N, 3) coordinates.
- `feat`: (N, C) features (e.g., intensity, return number, or RGB).
- `grid_size`: Voxel size for downsampling (recommended: 0.02m).
- `offset`: Cumulative sum of points in each batch sample.

### 2. Tiling & Memory Management
Since raw files are 5GB-35GB, they must be processed in **5m x 5m vertical columns/cubes**.
- **Preprocessing:** Use `laspy` to slice and downsample to a fixed density (e.g., 0.02m grid).
- **Inference:** Implement a sliding window or tiled approach to stitch predictions back into the original `.las` structure.

### 3. Classification Mapping
- `0`: Beach
- `1`: Cliff
- `2`: Vegetation

## Key Files
- `model.py`: Core PTv3 architecture and the `Point` data structure.
- `serialization/`: Space-filling curve implementations (Z-order, Hilbert) for point serialization.
- `plan.md`: The project roadmap and technical gap analysis.
- `README.md`: Official PTv3 instructions and benchmarks.
