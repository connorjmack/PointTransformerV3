# Implementation Roadmap: CoastalContrast Pipeline

## Current State Assessment

### What Exists

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| PTv3 Backbone | `model.py` | Complete | Full encoder-decoder architecture |
| Serialization | `serialization/` | Complete | Z-order and Hilbert curves |
| Basic Preprocessing | `prep_data.py` | Partial | Single-file tiling only |
| Tests | `tests/test_prep_data.py` | Basic | Needs expansion |

### What We Need to Build

```
coastal_contrast/
├── data/                          # Phase 1: Data Infrastructure
│   ├── __init__.py
│   ├── catalog.py                 # Survey index and metadata management
│   ├── registration.py            # ICP co-registration
│   ├── preprocess.py              # Enhanced preprocessing (normals, multi-res)
│   ├── correspondence.py          # Temporal correspondence mining
│   ├── temporal_dataset.py        # Temporal pair sampling
│   └── transforms.py              # Point cloud augmentations
├── models/                        # Phase 2: Model Components
│   ├── __init__.py
│   ├── coastal_contrast.py        # Pre-training wrapper
│   ├── projection_head.py         # Contrastive projection heads
│   ├── seg_head.py                # Segmentation head for fine-tuning
│   └── geometric_head.py          # Geometric prediction head (normals, curvature)
├── losses/                        # Phase 2: Loss Functions
│   ├── __init__.py
│   ├── temporal_contrast.py       # L_temporal
│   ├── spatial_contrast.py        # L_spatial
│   └── geometric.py               # L_geo (masked reconstruction)
├── training/                      # Phase 3: Training Infrastructure
│   ├── __init__.py
│   ├── pretrain.py                # Pre-training loop
│   ├── finetune.py                # Fine-tuning loop
│   └── utils.py                   # Checkpointing, logging, metrics
├── configs/                       # Configuration files
│   ├── pretrain/
│   │   └── coastal_contrast.yaml
│   └── finetune/
│       └── cliff_segmentation.yaml
└── scripts/                       # Entry point scripts
    ├── build_catalog.py           # Index all surveys
    ├── register_surveys.py        # Co-register to reference
    ├── mine_correspondences.py    # Extract temporal pairs
    ├── train_pretrain.py          # Launch pre-training
    ├── train_finetune.py          # Launch fine-tuning
    └── inference.py               # Run on new data
```

---

## Phase 1: Data Infrastructure (Priority: CRITICAL)

### 1.1 Survey Catalog System

**Purpose:** Manage 400+ LAS files with metadata for efficient temporal queries.

**File:** `coastal_contrast/data/catalog.py`

**Key Classes:**
```python
@dataclass
class SurveyMetadata:
    survey_id: str                    # Unique identifier
    file_path: Path                   # Path to LAS file
    date: datetime                    # Survey date
    platform: str                     # 'mobile', 'terrestrial', 'airborne'
    bounds: Tuple[float, ...]         # (min_x, min_y, max_x, max_y)
    point_count: int                  # Number of points
    resolution: float                 # Approximate point spacing
    is_registered: bool               # Has been co-registered
    reference_epoch: Optional[str]    # Reference survey ID if registered
    quality_score: float              # Data quality metric (0-1)

class SurveyCatalog:
    def __init__(self, catalog_path: Path)
    def add_survey(self, las_path: Path, metadata: dict) -> str
    def get_surveys_by_date_range(self, start: datetime, end: datetime) -> List[SurveyMetadata]
    def get_surveys_by_location(self, bounds: Tuple) -> List[SurveyMetadata]
    def get_temporal_pairs(self, delta_days: Tuple[int, int]) -> List[Tuple[str, str]]
    def get_overlapping_surveys(self, survey_id: str) -> List[SurveyMetadata]
    def save(self) / def load(self)
```

**Implementation Steps:**
1. Parse LAS file headers to extract metadata without loading full point clouds
2. Build spatial index (R-tree) for efficient location queries
3. Store catalog as SQLite database or Parquet for persistence
4. Add CLI for catalog operations (`python -m coastal_contrast.data.catalog add /path/to/surveys/`)

**Dependencies:** `laspy`, `rtree` (optional), `sqlite3` or `pandas`

---

### 1.2 Co-Registration Module

**Purpose:** Align temporal surveys to a common reference epoch.

**File:** `coastal_contrast/data/registration.py`

**Key Functions:**
```python
def select_reference_epoch(catalog: SurveyCatalog, criteria: str = 'quality') -> str:
    """Select best survey as reference based on quality, date, or coverage."""

def extract_stable_points(point_cloud: np.ndarray, method: str = 'planarity') -> np.ndarray:
    """Extract points likely to be stable (bedrock, infrastructure) for registration."""

def coarse_registration(source: np.ndarray, target: np.ndarray,
                        method: str = 'fpfh') -> np.ndarray:
    """Initial alignment using feature matching (FPFH descriptors)."""

def fine_registration_icp(source: np.ndarray, target: np.ndarray,
                          initial_transform: np.ndarray = None,
                          max_iterations: int = 100,
                          tolerance: float = 1e-6) -> Tuple[np.ndarray, float]:
    """Point-to-plane ICP for precise alignment. Returns (transform_4x4, rmse)."""

def register_survey(source_path: Path, reference_path: Path,
                    output_path: Path, method: str = 'icp') -> dict:
    """Full registration pipeline for a single survey."""

def batch_register(catalog: SurveyCatalog, reference_id: str,
                   output_dir: Path, n_workers: int = 4) -> None:
    """Register all surveys in catalog to reference epoch."""
```

**Implementation Strategy:**
1. Use Open3D for ICP (well-optimized, GPU-accelerated option available)
2. Extract stable points using planarity filter (flat surfaces = buildings, roads)
3. Two-stage registration: coarse (FPFH) → fine (point-to-plane ICP)
4. Store transformation matrices with catalog for reproducibility
5. Quality check: reject registrations with RMSE > threshold

**Dependencies:** `open3d`, `numpy`, `scipy`

**Estimated Complexity:** Medium-High (ICP tuning for coastal data)

---

### 1.3 Enhanced Preprocessing

**Purpose:** Extend `prep_data.py` for multi-resolution, normals, and additional features.

**File:** `coastal_contrast/data/preprocess.py`

**Key Functions:**
```python
def estimate_normals(coords: np.ndarray, k_neighbors: int = 30) -> np.ndarray:
    """Estimate surface normals using PCA on local neighborhoods."""

def estimate_curvature(coords: np.ndarray, normals: np.ndarray,
                       k_neighbors: int = 30) -> np.ndarray:
    """Estimate local curvature from normal variation."""

def compute_geometric_features(coords: np.ndarray) -> dict:
    """Compute full geometric feature set: normals, curvature, planarity, etc."""

def voxelize_with_features(coords: np.ndarray, features: np.ndarray,
                           labels: np.ndarray, grid_size: float,
                           aggregation: str = 'mean') -> Tuple:
    """Voxelize with feature aggregation (mean/max pooling)."""

def process_survey_multiresolution(las_path: Path, output_dir: Path,
                                   resolutions: List[float] = [0.02, 0.05, 0.1, 0.5],
                                   tile_size: float = 10.0) -> None:
    """Process single survey at multiple resolutions."""

def build_tile_index(output_dir: Path) -> pd.DataFrame:
    """Build index of all tiles with their spatial extents and resolutions."""
```

**Output Format (Extended):**
```python
{
    "coord": np.ndarray,           # (N, 3) XYZ
    "feat": np.ndarray,            # (N, F) features [intensity, return_num, ...]
    "normal": np.ndarray,          # (N, 3) surface normals
    "curvature": np.ndarray,       # (N, 1) local curvature
    "segment": np.ndarray,         # (N,) labels (if available)
    "grid_size": float,
    "survey_id": str,
    "tile_id": str,
    "bounds": Tuple[float, ...],
    "timestamp": str
}
```

**Dependencies:** `open3d` (for normals), `scikit-learn` (for PCA)

---

### 1.4 Temporal Correspondence Mining

**Purpose:** Find point correspondences across temporal surveys.

**File:** `coastal_contrast/data/correspondence.py`

**Key Classes and Functions:**
```python
@dataclass
class Correspondence:
    source_idx: int                    # Index in source point cloud
    target_idx: int                    # Index in target point cloud
    distance: float                    # Spatial distance
    normal_agreement: float            # Dot product of normals
    confidence: float                  # Combined confidence score
    is_change_candidate: bool          # Flagged as potential change

class CorrespondenceMiner:
    def __init__(self,
                 max_distance: float = 0.5,      # Max correspondence distance (m)
                 normal_threshold: float = 0.8,   # Min normal agreement
                 k_neighbors: int = 5,            # Neighbors to consider
                 change_threshold: float = 0.1):  # Confidence below = change

    def mine_correspondences(self,
                             source_tile: dict,
                             target_tile: dict) -> List[Correspondence]:
        """Find correspondences between two co-registered tiles."""

    def mine_temporal_stack(self,
                            tile_id: str,
                            survey_ids: List[str],
                            catalog: SurveyCatalog) -> dict:
        """Mine correspondences across all surveys for a spatial tile."""

    def save_correspondences(self, output_path: Path) -> None
    def load_correspondences(self, input_path: Path) -> None

def build_correspondence_graph(correspondences: List[Correspondence],
                               min_confidence: float = 0.5) -> nx.Graph:
    """Build graph of point correspondences for transitive matching."""

def filter_change_candidates(correspondences: List[Correspondence],
                             method: str = 'statistical') -> List[Correspondence]:
    """Identify correspondences likely representing real change vs noise."""
```

**Mining Strategy:**
1. For each tile location, load all temporal instances
2. Build KD-tree on target survey
3. For each source point, find k-nearest in target within radius
4. Compute confidence: `conf = (1 - dist/max_dist) * max(0, n1·n2)`
5. Flag low-confidence as change candidates
6. Store as HDF5 or Arrow format for efficient loading

**Output Format:**
```python
# Per tile-pair correspondence file
{
    "source_survey": str,
    "target_survey": str,
    "correspondences": np.ndarray,  # (M, 4) [src_idx, tgt_idx, dist, conf]
    "change_mask": np.ndarray       # (M,) bool
}
```

**Dependencies:** `scipy` (KDTree), `h5py` or `pyarrow`

---

## Phase 2: Dataset and DataLoaders

### 2.1 Temporal Pair Dataset

**Purpose:** Sample temporal pairs for contrastive pre-training.

**File:** `coastal_contrast/data/temporal_dataset.py`

```python
class TemporalPairDataset(Dataset):
    def __init__(self,
                 catalog: SurveyCatalog,
                 correspondence_dir: Path,
                 resolutions: List[float] = [0.02],
                 temporal_intervals: List[int] = [7, 30, 90, 365],  # days
                 points_per_tile: int = 50000,
                 transform: Optional[Callable] = None):

    def __getitem__(self, idx) -> dict:
        """
        Returns:
            {
                "source_coord": Tensor,
                "source_feat": Tensor,
                "source_normal": Tensor,
                "target_coord": Tensor,
                "target_feat": Tensor,
                "target_normal": Tensor,
                "correspondences": Tensor,  # (M, 2) indices
                "confidence": Tensor,       # (M,) confidence scores
                "delta_days": int,          # Time between surveys
            }
        """

    def __len__(self) -> int

    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        """Custom collate for variable-size point clouds."""
```

### 2.2 Point Cloud Transforms

**File:** `coastal_contrast/data/transforms.py`

```python
class Compose:
    """Compose multiple transforms."""

class RandomRotation:
    """Random rotation around Z-axis (or full 3D)."""

class RandomFlip:
    """Random horizontal flip."""

class RandomScale:
    """Random uniform scaling."""

class RandomJitter:
    """Add Gaussian noise to coordinates."""

class RandomDropout:
    """Randomly drop points."""

class ElasticDistortion:
    """Elastic deformation for regularization."""

class NormalizeCoordinates:
    """Center and scale coordinates."""

class VoxelSubsample:
    """Random voxel-based subsampling."""
```

---

## Phase 3: Model Components

### 3.1 Projection Head

**File:** `coastal_contrast/models/projection_head.py`

```python
class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 256,
                 out_channels: int = 128,
                 num_layers: int = 2):

    def forward(self, x: Tensor) -> Tensor:
        """Project features to contrastive embedding space."""
```

### 3.2 Geometric Prediction Head

**File:** `coastal_contrast/models/geometric_head.py`

```python
class GeometricHead(nn.Module):
    """Head for predicting geometric properties (normals, curvature, occupancy)."""

    def __init__(self,
                 in_channels: int,
                 predict_normals: bool = True,
                 predict_curvature: bool = True,
                 predict_occupancy: bool = True,
                 occupancy_grid_size: float = 0.1,
                 occupancy_range: int = 3):  # 3x3x3 occupancy grid

    def forward(self, features: Tensor, coords: Tensor) -> dict:
        """
        Returns:
            {
                "pred_normals": Tensor,    # (N, 3)
                "pred_curvature": Tensor,  # (N, 1)
                "pred_occupancy": Tensor   # (N, 27) for 3x3x3 grid
            }
        """
```

### 3.3 CoastalContrast Model

**File:** `coastal_contrast/models/coastal_contrast.py`

```python
class CoastalContrast(nn.Module):
    """
    Self-supervised pre-training model combining:
    - PTv3 backbone (encoder only for pre-training)
    - Temporal contrastive learning
    - Spatial contrastive learning
    - Geometric masked prediction
    """

    def __init__(self,
                 backbone: PointTransformerV3,
                 projection_dim: int = 128,
                 temperature: float = 0.1,
                 mask_ratio: float = 0.6,
                 enable_temporal: bool = True,
                 enable_spatial: bool = True,
                 enable_geometric: bool = True):

    def forward(self, batch: dict) -> dict:
        """
        Args:
            batch: Dict with source/target point clouds and correspondences

        Returns:
            {
                "source_features": Tensor,
                "target_features": Tensor,
                "source_projections": Tensor,
                "target_projections": Tensor,
                "geometric_predictions": dict,
                "loss_temporal": Tensor,
                "loss_spatial": Tensor,
                "loss_geometric": Tensor,
                "loss_total": Tensor
            }
        """

    def encode(self, data_dict: dict) -> Tensor:
        """Encode a single point cloud (for inference)."""
```

---

## Phase 4: Loss Functions

### 4.1 Temporal Contrastive Loss

**File:** `coastal_contrast/losses/temporal_contrast.py`

```python
class TemporalContrastiveLoss(nn.Module):
    """
    InfoNCE loss for temporal correspondences.

    For each point in source, its corresponding point in target is positive,
    all other target points are negatives.
    """

    def __init__(self,
                 temperature: float = 0.1,
                 use_confidence_weighting: bool = True):

    def forward(self,
                source_proj: Tensor,      # (N, D) source projections
                target_proj: Tensor,      # (M, D) target projections
                correspondences: Tensor,  # (K, 2) [src_idx, tgt_idx]
                confidence: Tensor        # (K,) confidence weights
               ) -> Tensor:
```

### 4.2 Geometric Loss

**File:** `coastal_contrast/losses/geometric.py`

```python
class GeometricLoss(nn.Module):
    """
    Combined loss for geometric predictions.
    """

    def __init__(self,
                 normal_weight: float = 1.0,
                 curvature_weight: float = 0.5,
                 occupancy_weight: float = 0.5):

    def forward(self,
                pred_normals: Tensor,
                gt_normals: Tensor,
                pred_curvature: Tensor,
                gt_curvature: Tensor,
                pred_occupancy: Tensor,
                gt_occupancy: Tensor,
                mask: Tensor              # (N,) which points were masked
               ) -> dict:
```

---

## Phase 5: Training Infrastructure

### 5.1 Pre-Training Loop

**File:** `coastal_contrast/training/pretrain.py`

```python
class PreTrainer:
    def __init__(self,
                 model: CoastalContrast,
                 train_dataset: TemporalPairDataset,
                 val_dataset: TemporalPairDataset,
                 config: dict):

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch, return metrics."""

    def validate(self) -> dict:
        """Run validation, return metrics."""

    def save_checkpoint(self, path: Path, epoch: int) -> None

    def load_checkpoint(self, path: Path) -> int:
        """Load checkpoint, return epoch number."""

    def train(self, num_epochs: int, checkpoint_dir: Path) -> None:
        """Full training loop with logging and checkpointing."""
```

### 5.2 Fine-Tuning Loop

**File:** `coastal_contrast/training/finetune.py`

```python
class FineTuner:
    def __init__(self,
                 backbone: PointTransformerV3,
                 pretrained_weights: Path,
                 num_classes: int,
                 freeze_strategy: str = 'progressive'):  # 'none', 'backbone', 'progressive'

    def train(self, ...) -> None
```

---

## Implementation Order

### Sprint 1: Foundation (Week 1-2)
1. **Project structure setup** - Create directory layout, `__init__.py` files
2. **Survey catalog** - Basic metadata management and persistence
3. **Enhanced preprocessing** - Normal estimation, multi-resolution tiling
4. **Tests** - Unit tests for each component

### Sprint 2: Registration & Correspondence (Week 2-3)
5. **Co-registration module** - ICP implementation with Open3D
6. **Correspondence mining** - KD-tree based temporal matching
7. **Integration tests** - End-to-end preprocessing pipeline

### Sprint 3: Dataset & Model (Week 3-4)
8. **Temporal pair dataset** - DataLoader with correspondence sampling
9. **Projection and geometric heads** - Simple MLP heads
10. **CoastalContrast wrapper** - Integrate backbone with heads

### Sprint 4: Training (Week 4-5)
11. **Loss functions** - Temporal, spatial, geometric losses
12. **Training loop** - Pre-training with checkpointing
13. **Validation and metrics** - Loss curves, feature quality checks

### Sprint 5: Validation (Week 5-6)
14. **Small-scale test** - Pre-train on subset (1 year data)
15. **Feature visualization** - t-SNE/UMAP of learned features
16. **Downstream evaluation** - Linear probe on segmentation

---

## Hardware Requirements

| Phase | GPU Memory | Storage | CPU Cores |
|-------|------------|---------|-----------|
| Preprocessing | 0 | 50TB | 16+ |
| Correspondence Mining | 8GB | 10TB | 32+ |
| Pre-training | 40GB+ (A100) | 1TB | 8+ |
| Fine-tuning | 24GB | 100GB | 8+ |

---

## Dependencies

```txt
# Core
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0

# Point cloud processing
laspy>=2.4.0
open3d>=0.17.0
torch-scatter>=2.1.0
spconv-cu118>=2.3.0  # CUDA version dependent

# PTv3 specific
flash-attn>=2.0.0    # Optional, requires CUDA 11.6+
addict>=2.4.0
timm>=0.9.0

# Data management
pandas>=2.0.0
h5py>=3.8.0
pyarrow>=12.0.0

# Training
tensorboard>=2.12.0
wandb>=0.15.0        # Optional
pyyaml>=6.0

# Testing
pytest>=7.3.0
pytest-cov>=4.0.0
```

---

## Quick Start (After Implementation)

```bash
# 1. Build survey catalog
python -m coastal_contrast.scripts.build_catalog \
    --input-dir /data/san_diego_lidar/ \
    --output catalog.db

# 2. Co-register surveys
python -m coastal_contrast.scripts.register_surveys \
    --catalog catalog.db \
    --reference-id survey_2020_01_01 \
    --output-dir /data/registered/

# 3. Mine temporal correspondences
python -m coastal_contrast.scripts.mine_correspondences \
    --catalog catalog.db \
    --registered-dir /data/registered/ \
    --output-dir /data/correspondences/

# 4. Pre-train CoastalContrast
python -m coastal_contrast.scripts.train_pretrain \
    --config configs/pretrain/coastal_contrast.yaml \
    --catalog catalog.db \
    --correspondence-dir /data/correspondences/ \
    --output-dir /models/pretrain/

# 5. Fine-tune for segmentation
python -m coastal_contrast.scripts.train_finetune \
    --config configs/finetune/cliff_segmentation.yaml \
    --pretrained /models/pretrain/best.pth \
    --train-data /data/labeled/train/ \
    --output-dir /models/finetune/
```

---

## Next Steps

**Immediate action:** Which component should we implement first?

Recommended order:
1. **Survey catalog** - Foundation for everything else
2. **Enhanced preprocessing** - Need multi-resolution tiles before anything else
3. **Correspondence mining** - Core of temporal contrastive learning
4. **Temporal dataset** - Required for training
5. **Loss functions + model** - The actual CoastalContrast framework

Would you like me to start implementing the survey catalog system?
