# Temporal Contrastive Learning for Coastal LiDAR Classification
## A Self-Supervised Approach to Geomorphological Feature Segmentation

**Principal Investigator:** [Your Name]
**Institution:** [Your Institution]
**Version:** 1.0
**Date:** January 2026

---

## Executive Summary

This research plan proposes a novel self-supervised learning framework for coastal LiDAR classification that exploits **8 years of semi-weekly temporal revisits** across San Diego County and statewide California flight data. The central hypothesis is that **temporal consistency in geomorphological features provides a powerful supervisory signal** for learning robust point cloud representations without manual annotation.

Unlike existing approaches that adapt urban/indoor models to coastal domains, we propose learning coastal-specific representations *ab initio* from temporal correspondence. A cliff face scanned on different dates should produce similar embeddings—unless actual erosion has occurred. This temporal invariance constraint, combined with spatial contrastive objectives, creates millions of free training pairs from unlabeled data.

**Key Innovation:** We move self-supervised pre-training from an auxiliary technique to the **centerpiece** of the pipeline, enabled by the unprecedented scale of temporal LiDAR data available.

---

## 1. Scientific Background and Motivation

### 1.1 The Problem

Current deep learning approaches for point cloud segmentation face a fundamental limitation: they require extensive labeled training data. For autonomous driving (nuScenes, Waymo) and indoor scenes (ScanNet), large annotated datasets exist. For coastal geomorphology—cliff faces, riprap, seawalls, beach morphology—**no such datasets exist**.

The conventional solution involves:
1. Pre-training on urban/driving datasets
2. Transfer learning with limited coastal labels
3. Active learning to minimize annotation costs

This approach suffers from:
- **Semantic domain gap**: Urban classes (cars, buildings, pedestrians) share no conceptual overlap with coastal classes (eroding sandstone, beach cusps, cliff vegetation)
- **Geometric domain gap**: Spinning automotive LiDAR produces radial point distributions fundamentally different from airborne/terrestrial coastal surveys
- **Scale mismatch**: Indoor datasets operate at sub-meter scales; coastal monitoring spans kilometers

### 1.2 The Opportunity

Your dataset represents an extraordinary opportunity:

| Dataset Component | Scale | Temporal Density |
|-------------------|-------|------------------|
| San Diego County (terrestrial/mobile) | 8 years | Semi-weekly (~400 surveys) |
| California Statewide (airborne) | Multiple years | Annual/biannual flights |

This provides:
- **Millions of temporal correspondence pairs** from co-registered revisits
- **Natural augmentation** through seasonal variation, tide levels, vegetation phenology
- **Implicit change detection labels** where differences exceed noise thresholds
- **Multi-resolution coverage** from cm-scale mobile to m-scale airborne

### 1.3 Central Hypothesis

> **H1 (Primary):** Temporal contrastive learning on multi-year coastal LiDAR surveys produces feature representations that transfer more effectively to coastal segmentation tasks than representations learned from urban/indoor pre-training.

> **H2 (Secondary):** The same temporal framework can simultaneously learn a change detection model that identifies genuine geomorphological change versus sensor noise, further improving segmentation quality.

> **H3 (Tertiary):** Representations learned from high-density terrestrial surveys (San Diego) transfer effectively to lower-density airborne surveys (statewide), enabling a unified coastal classification framework.

---

## 2. Technical Approach

### 2.1 Architecture Selection: Point Transformer V3

PTv3 is selected as the backbone architecture for several reasons:

1. **Sensor-agnostic design**: Unlike Cylinder3D or SphereFormer, PTv3 makes no assumptions about radial LiDAR geometry
2. **Scalable attention**: Serialized attention via space-filling curves handles million-point clouds
3. **Memory efficiency**: Flash Attention integration provides 10× memory reduction
4. **State-of-the-art performance**: 73.4% mIoU on SemanticKITTI, 80.3% on nuScenes
5. **Available codebase**: Pointcept framework with modular training pipeline

### 2.2 Temporal Contrastive Pre-Training Framework

We propose **CoastalContrast**, a self-supervised framework with three contrastive objectives:

#### 2.2.1 Temporal Consistency Loss (L_temporal)

For co-registered point clouds P_t and P_{t+Δt} from the same location at different times:

```
L_temporal = -log(exp(sim(z_i^t, z_i^{t+Δt})/τ) / Σ_j exp(sim(z_i^t, z_j^{t+Δt})/τ))
```

Where:
- `z_i^t` is the feature embedding of point i at time t
- `sim(·,·)` is cosine similarity
- `τ` is temperature parameter
- The denominator includes both positive (corresponding) and negative (non-corresponding) points

**Key insight**: Points that *should* be the same (stable geology) are positive pairs. Points with genuine change become hard negatives that the model must distinguish.

#### 2.2.2 Spatial Contrastive Loss (L_spatial)

Following PointContrast, we enforce consistency across different spatial contexts:

```
L_spatial = -log(exp(sim(z_i, z_i')/τ) / Σ_j exp(sim(z_i, z_j')/τ))
```

Where z_i and z_i' are embeddings of the same point from overlapping tiles with different spatial contexts.

#### 2.2.3 Geometric Reconstruction Loss (L_geo)

Following GeoMAE, we mask portions of the point cloud and predict geometric properties:

```
L_geo = ||n̂_i - n_i||² + ||κ̂_i - κ_i||² + ||c_i^{occ} - ĉ_i^{occ}||²
```

Where:
- n_i = surface normal at point i
- κ_i = local curvature
- c_i^{occ} = occupancy in surrounding voxels

This teaches the model to understand coastal geometry (cliff angles, beach slopes, vegetation structure).

#### 2.2.4 Combined Objective

```
L_pretrain = λ_1 · L_temporal + λ_2 · L_spatial + λ_3 · L_geo
```

With λ weights determined through ablation studies.

### 2.3 Temporal Correspondence Mining

A critical technical challenge is establishing point correspondences across time without manual registration.

**Proposed Pipeline:**

1. **Coarse alignment**: ICP (Iterative Closest Point) on stable reference points (survey monuments, bedrock outcrops)
2. **Dense correspondence**: For each point p_i^t, find k-nearest neighbors in P_{t+Δt} within radius r
3. **Confidence weighting**: Weight correspondences by:
   - Spatial proximity (closer = higher weight)
   - Local geometry similarity (surface normal agreement)
   - Intensity consistency (if available)
4. **Change filtering**: Flag correspondences where feature distance exceeds threshold σ as potential changes (not positive pairs)

```python
def mine_temporal_correspondences(P_t, P_t_delta, k=5, r=0.5, sigma=0.1):
    """
    Mine temporal correspondences between two co-registered point clouds.

    Returns:
        correspondences: List of (idx_t, idx_t_delta, confidence) tuples
        change_candidates: Points where temporal features diverge significantly
    """
    # Build KD-tree for efficient neighbor search
    tree = KDTree(P_t_delta.coord)

    correspondences = []
    change_candidates = []

    for i, p in enumerate(P_t.coord):
        # Find k-nearest neighbors within radius r
        dists, indices = tree.query(p, k=k, distance_upper_bound=r)

        valid = dists < r
        if not valid.any():
            continue

        # Compute confidence based on distance and normal agreement
        for j, d in zip(indices[valid], dists[valid]):
            normal_sim = np.dot(P_t.normal[i], P_t_delta.normal[j])
            confidence = (1 - d/r) * max(0, normal_sim)

            if confidence > sigma:
                correspondences.append((i, j, confidence))
            else:
                change_candidates.append((i, j, confidence))

    return correspondences, change_candidates
```

### 2.4 Multi-Scale Architecture for Cross-Resolution Transfer

To handle the resolution gap between terrestrial (cm-scale) and airborne (m-scale) surveys:

**Hierarchical Pre-Training Strategy:**

1. **Stage 1: High-Resolution Foundation** (San Diego mobile LiDAR)
   - Pre-train at native resolution (0.01-0.02m voxel)
   - Learn fine geometric details: cliff texture, riprap edges, vegetation structure

2. **Stage 2: Multi-Resolution Adaptation**
   - Progressively downsample to 0.05m, 0.1m, 0.5m, 1.0m voxels
   - Apply temporal contrastive learning at each resolution
   - Share encoder weights across resolutions (multi-scale consistency)

3. **Stage 3: Airborne Integration** (Statewide data)
   - Transfer pre-trained weights to airborne data (typically 0.5-2m resolution)
   - Fine-tune with airborne-specific augmentations (flight line artifacts, varying density)

### 2.5 Downstream Task Heads

The pre-trained backbone supports multiple downstream tasks:

**A. Semantic Segmentation Head**
```
PTv3_Encoder → PTv3_Decoder → Linear(512, num_classes) → Softmax
```

Classes for coastal domain:
- Beach (bare sand/gravel)
- Cliff face (exposed geology)
- Cliff vegetation (rooted on cliff)
- Upland vegetation (above cliff)
- Riprap/Armor
- Seawall/Structure
- Water/Wet sand
- Infrastructure (roads, buildings)

**B. Change Detection Head**
```
[z_i^t; z_i^{t+Δt}; |z_i^t - z_i^{t+Δt}|] → MLP → {no_change, erosion, accretion, anthropogenic}
```

The temporal pre-training naturally supports change detection—the contrastive learning teaches what "same" looks like, so deviations indicate change.

**C. Erosion Rate Regression Head**
```
Temporal_Feature_Sequence → Transformer → Volumetric_Change_Rate
```

Using multiple time steps to predict erosion velocity.

---

## 3. Experimental Design

### 3.1 Dataset Construction

#### 3.1.1 San Diego County Dataset (Primary)

| Attribute | Specification |
|-----------|--------------|
| Coverage | Del Mar to Imperial Beach coastline |
| Duration | 8 years (2018-2026) |
| Frequency | Semi-weekly (26-52 surveys/year) |
| Total surveys | ~400 |
| Points per survey | 10⁸ - 10⁹ |
| Resolution | 0.01-0.05m point spacing |
| Platform | Mobile/terrestrial laser scanning |
| Attributes | XYZ, intensity, return number |

**Preprocessing Pipeline:**
1. Quality filtering (remove noise, water returns)
2. Co-registration to reference epoch using ICP
3. Tiling into 10m × 10m × 50m columns
4. Voxelization at target resolution
5. Surface normal and curvature computation
6. Temporal correspondence mining

#### 3.1.2 California Statewide Dataset (Secondary)

| Attribute | Specification |
|-----------|--------------|
| Coverage | California coastline (1,100 miles) |
| Duration | Multiple years |
| Frequency | Annual/biannual flights |
| Resolution | 0.5-2m point spacing |
| Platform | Airborne LiDAR |
| Attributes | XYZ, intensity, classification |

**Use Cases:**
- Cross-resolution transfer validation
- Statewide model generalization testing
- Comparison with high-resolution San Diego results

### 3.2 Experimental Conditions

#### Experiment 1: Pre-Training Strategy Comparison

Compare CoastalContrast against:

| Condition | Description |
|-----------|-------------|
| **Baseline A** | Random initialization, train from scratch on labeled data |
| **Baseline B** | nuScenes pre-training (standard transfer) |
| **Baseline C** | DALES pre-training (closest aerial domain) |
| **Baseline D** | GeoMAE-only pre-training (no temporal) |
| **Baseline E** | PointContrast pre-training (spatial-only) |
| **CoastalContrast** | Full temporal + spatial + geometric pre-training |

**Metric**: mIoU on held-out labeled test set with varying amounts of fine-tuning labels (1%, 5%, 10%, 25%, 50%, 100%)

**Hypothesis**: CoastalContrast achieves equivalent performance to baselines with 5-10× less labeled data.

#### Experiment 2: Temporal Density Ablation

How much temporal data is needed?

| Condition | Temporal Coverage |
|-----------|-------------------|
| T-1 | 1 year (52 surveys) |
| T-2 | 2 years (104 surveys) |
| T-4 | 4 years (208 surveys) |
| T-8 | 8 years (416 surveys) |

**Metric**: Pre-training loss convergence and downstream task performance

**Hypothesis**: Performance improves logarithmically with temporal coverage; diminishing returns after ~4 years.

#### Experiment 3: Temporal Interval Effects

Optimal time delta between contrastive pairs?

| Condition | Time Delta |
|-----------|------------|
| Δt-1w | 1 week |
| Δt-1m | 1 month |
| Δt-3m | 3 months (seasonal) |
| Δt-1y | 1 year (annual) |
| Δt-mixed | Random sampling from all intervals |

**Hypothesis**: Mixed intervals perform best by learning both short-term stability and long-term invariance.

#### Experiment 4: Cross-Resolution Transfer

Does pre-training transfer across resolutions?

| Source | Target | Domain Gap |
|--------|--------|------------|
| Mobile (0.02m) | Mobile (0.02m) | None (control) |
| Mobile (0.02m) | Airborne (0.5m) | Resolution only |
| Mobile (0.02m) | Airborne different region | Resolution + geography |
| Airborne (0.5m) | Airborne (0.5m) | None (control) |

**Metric**: mIoU degradation relative to in-domain pre-training

#### Experiment 5: Change Detection Emergence

Does temporal pre-training naturally enable change detection?

Compare:
- Binary classifier trained on labeled change data
- Zero-shot change detection via embedding distance threshold
- Few-shot change detection with 10 labeled examples

**Hypothesis**: Zero-shot change detection achieves >70% of supervised performance, demonstrating that temporal contrastive learning implicitly learns change.

### 3.3 Evaluation Metrics

**Segmentation Performance:**
- Mean Intersection over Union (mIoU)
- Per-class IoU (especially for challenging classes: cliff vegetation, riprap)
- Overall accuracy (OA)
- Cohen's Kappa

**Label Efficiency:**
- Performance vs. labeled data percentage curves
- Data efficiency ratio: (mIoU_pretrained / mIoU_scratch) at each label fraction

**Change Detection:**
- Precision/Recall/F1 for change categories
- Volumetric error (predicted vs. measured erosion)
- False alarm rate on stable areas

**Computational:**
- Pre-training wall-clock time
- Fine-tuning convergence epochs
- Inference throughput (points/second)

### 3.4 Annotation Strategy

Despite the self-supervised focus, evaluation requires labeled ground truth.

**Annotation Protocol:**
1. Stratified sampling: Select tiles covering diverse coastal morphologies
2. Multi-annotator labeling: 3 geomorphology experts per tile
3. Quality control: Inter-annotator agreement (Fleiss' Kappa) > 0.8 required
4. Class definitions: Standardized taxonomy with visual examples

**Target Annotation Budget:**
| Phase | Tiles | Points | Purpose |
|-------|-------|--------|---------|
| Validation | 100 | ~10M | Hyperparameter tuning |
| Test | 200 | ~20M | Final evaluation |
| Active learning pool | 1000 | ~100M | Iterative labeling |

**Annotation Tool**: Segment.ai or custom Open3D-based interface with AI-assisted suggestions from pre-trained model.

---

## 4. Implementation Plan

### 4.1 Phase 1: Infrastructure and Data Pipeline (Months 1-2)

**Deliverables:**
- [ ] Data ingestion pipeline for 400+ LAS files
- [ ] Co-registration framework using ICP
- [ ] Temporal correspondence mining system
- [ ] Tiling and voxelization at multiple resolutions
- [ ] Data loader for PTv3 training

**Technical Components:**

```python
# Core data pipeline structure
coastal_dataset/
├── raw/                    # Original LAS files
│   ├── san_diego/
│   │   ├── 2018/
│   │   │   ├── 2018-01-05.las
│   │   │   ├── 2018-01-12.las
│   │   │   └── ...
│   │   └── ...
│   └── statewide/
├── processed/
│   ├── aligned/            # Co-registered point clouds
│   ├── tiles/              # 10m x 10m x 50m tiles
│   │   ├── res_0.02/       # High-resolution tiles
│   │   ├── res_0.10/       # Medium-resolution tiles
│   │   └── res_0.50/       # Low-resolution tiles
│   └── correspondences/    # Temporal correspondence files
├── annotations/
│   ├── train/
│   ├── val/
│   └── test/
└── metadata/
    ├── survey_index.csv    # Survey dates, extents, quality
    └── tile_index.csv      # Tile locations, temporal coverage
```

### 4.2 Phase 2: CoastalContrast Pre-Training (Months 2-4)

**Deliverables:**
- [ ] Temporal contrastive loss implementation
- [ ] Multi-resolution training framework
- [ ] Pre-training on full San Diego dataset
- [ ] Checkpoint management and evaluation hooks

**Training Configuration:**

```yaml
# config/pretrain_coastal_contrast.yaml
model:
  name: PointTransformerV3
  in_channels: 6  # XYZ + intensity + return_num + computed_features
  enc_depths: [2, 2, 2, 6, 2]
  enc_channels: [32, 64, 128, 256, 512]
  enable_flash: true

pretrain:
  method: coastal_contrast
  objectives:
    temporal:
      weight: 1.0
      temperature: 0.1
      max_correspondences_per_point: 5
    spatial:
      weight: 0.5
      temperature: 0.1
      overlap_ratio: 0.3
    geometric:
      weight: 0.5
      mask_ratio: 0.6
      predict: [normal, curvature, occupancy]

data:
  train_resolutions: [0.02, 0.05, 0.10]
  tile_size: [10, 10, 50]  # meters
  temporal_intervals: [7, 30, 90, 365]  # days

optimizer:
  name: AdamW
  lr: 0.001
  weight_decay: 0.05

scheduler:
  name: CosineAnnealingWarmRestarts
  T_0: 50
  T_mult: 2

training:
  epochs: 200
  batch_size: 16
  num_workers: 8
  mixed_precision: true
```

### 4.3 Phase 3: Annotation and Supervised Fine-Tuning (Months 4-5)

**Deliverables:**
- [ ] Annotation interface and guidelines
- [ ] 100 validation tiles annotated
- [ ] 200 test tiles annotated
- [ ] Fine-tuning pipeline with active learning

**Active Learning Loop:**

```python
def active_learning_iteration(model, unlabeled_pool, budget):
    """Select most informative samples for annotation."""

    # 1. Compute uncertainty via MC Dropout
    uncertainties = compute_mc_dropout_entropy(model, unlabeled_pool, n_forward=20)

    # 2. Compute diversity via k-center greedy
    features = extract_features(model, unlabeled_pool)
    diversity_scores = k_center_greedy(features, budget)

    # 3. Combined acquisition score
    acquisition_scores = uncertainties * diversity_scores

    # 4. Select top samples
    selected_indices = np.argsort(acquisition_scores)[-budget:]

    return unlabeled_pool[selected_indices]
```

### 4.4 Phase 4: Experiments and Analysis (Months 5-7)

**Deliverables:**
- [ ] Complete experimental matrix (Experiments 1-5)
- [ ] Statistical significance testing
- [ ] Ablation studies
- [ ] Error analysis and failure case documentation

### 4.5 Phase 5: Statewide Deployment and Validation (Months 7-9)

**Deliverables:**
- [ ] Transfer learning to airborne statewide data
- [ ] Cross-region validation
- [ ] Production inference pipeline
- [ ] Documentation and model release

### 4.6 Phase 6: Publication and Dissemination (Months 9-12)

**Deliverables:**
- [ ] Primary manuscript (target: CVPR or ICCV)
- [ ] Technical report for coastal management stakeholders
- [ ] Open-source code release
- [ ] Pre-trained model weights on HuggingFace

---

## 5. Expected Contributions

### 5.1 Scientific Contributions

1. **First large-scale temporal contrastive pre-training for coastal LiDAR**: Demonstrating that temporal revisits provide superior pre-training signal compared to synthetic augmentations

2. **Coastal geomorphology benchmark**: Publishing a curated labeled dataset for cliff/beach/vegetation segmentation (first of its kind)

3. **Cross-resolution transfer methodology**: Framework for learning from high-resolution data and deploying to lower-resolution aerial surveys

4. **Implicit change detection**: Demonstrating that contrastive learning encodes change detection capability without explicit supervision

### 5.2 Practical Contributions

1. **Production segmentation model**: Pre-trained PTv3 weights specialized for coastal LiDAR, available for coastal managers and researchers

2. **Data pipeline toolkit**: Open-source code for processing temporal LiDAR stacks

3. **Annotation efficiency**: Demonstrating 5-10× reduction in labeling requirements through self-supervised pre-training

### 5.3 Methodological Contributions

1. **Temporal correspondence mining**: Algorithms for establishing point correspondences across LiDAR revisits with varying conditions

2. **Multi-resolution curriculum**: Training strategy that bridges terrestrial and airborne LiDAR scales

---

## 6. Risk Assessment and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Co-registration errors propagate to pre-training | Medium | High | Multi-stage registration with quality thresholds; correspondence confidence weighting |
| Temporal contrastive learning collapses (all embeddings identical) | Low | Critical | Implement loss monitoring; add regularization; ensure sufficient hard negatives |
| Insufficient labeled data for meaningful evaluation | Medium | High | Prioritize annotation early; establish partnerships with coastal agencies |
| Computational cost of pre-training exceeds resources | Medium | Medium | Start with subset experiments; use mixed-precision; consider cloud computing |
| Change detection signal dominates stable features | Low | Medium | Stratified sampling to ensure stable areas represented; tune loss weights |
| Airborne transfer fails due to domain gap | Medium | Medium | Intermediate resolution fine-tuning; domain adaptation techniques |

---

## 7. Resource Requirements

### 7.1 Computational

| Resource | Specification | Purpose |
|----------|--------------|---------|
| GPU cluster | 4× NVIDIA A100 (80GB) | Pre-training |
| Storage | 50TB SSD | Raw and processed LiDAR |
| CPU nodes | 64-core workstations | Preprocessing, correspondence mining |

**Estimated compute budget:**
- Pre-training: ~2000 GPU-hours
- Fine-tuning experiments: ~500 GPU-hours
- Inference and evaluation: ~200 GPU-hours

### 7.2 Data

| Dataset | Status | Access |
|---------|--------|--------|
| San Diego mobile LiDAR | Available | Local |
| California statewide airborne | Available | NOAA Digital Coast / State GIS |
| Validation labels | To be created | Expert annotation |

### 7.3 Personnel

| Role | FTE | Responsibility |
|------|-----|----------------|
| ML Research Lead | 1.0 | Architecture, training, experiments |
| Data Engineer | 0.5 | Pipeline, preprocessing, infrastructure |
| Geomorphology Expert | 0.25 | Annotation oversight, domain guidance |
| Annotators | 0.5 | Manual labeling |

---

## 8. Timeline Summary

```
Month 1-2:   [████████] Infrastructure & Data Pipeline
Month 2-4:   [████████████████] CoastalContrast Pre-Training
Month 4-5:   [████████] Annotation & Fine-Tuning
Month 5-7:   [████████████] Experiments & Analysis
Month 7-9:   [████████] Statewide Deployment
Month 9-12:  [████████████] Publication & Release
```

---

## 9. Success Criteria

### 9.1 Primary Success Metrics

| Metric | Target | Stretch |
|--------|--------|---------|
| mIoU with 10% labels (vs. scratch) | +15 points | +25 points |
| mIoU with 100% labels | 75% | 80% |
| Label efficiency (same mIoU, less data) | 5× | 10× |
| Zero-shot change detection F1 | 0.70 | 0.80 |
| Airborne transfer degradation | <10% mIoU loss | <5% mIoU loss |

### 9.2 Publication Target

- **Primary venue**: CVPR, ICCV, or ECCV (computer vision)
- **Secondary venue**: Remote Sensing of Environment, ISPRS (geoscience)
- **Tertiary venue**: Earth Science Reviews (interdisciplinary impact)

---

## 10. Broader Impact

### 10.1 Scientific Impact

This work opens a new paradigm for geoscience deep learning: **learning from temporal revisits rather than manual labels**. The methodology generalizes to any domain with repeated surveys:
- Glacier monitoring
- Forest inventory
- Urban change detection
- Infrastructure inspection
- Archaeological surveys

### 10.2 Societal Impact

Improved coastal classification directly supports:
- **Climate adaptation planning**: Better erosion monitoring informs setback policies
- **Hazard assessment**: Cliff collapse prediction for public safety
- **Ecosystem management**: Vegetation tracking for habitat conservation
- **Infrastructure protection**: Seawall and riprap condition assessment

### 10.3 Ethical Considerations

- **Data privacy**: Coastal LiDAR may capture private property; anonymization protocols for public release
- **Model misuse**: Ensure model limitations are documented to prevent overconfident predictions in critical decisions
- **Equitable access**: Publish pre-trained weights freely to democratize coastal monitoring

---

## Appendix A: Literature References

### Self-Supervised Point Cloud Learning
1. Xie, S., et al. (2020). PointContrast: Unsupervised pre-training for 3D point cloud understanding. ECCV.
2. Zhang, Z., et al. (2021). DepthContrast: Self-supervised pre-training for 3D point cloud representation learning. ICCV.
3. Pang, Y., et al. (2022). Masked autoencoders for point cloud self-supervised learning. ECCV.
4. Tian, X., et al. (2023). GeoMAE: Masked geometric target prediction for self-supervised point cloud pre-training. CVPR.

### Point Cloud Segmentation Architectures
5. Wu, X., et al. (2024). Point Transformer V3: Simpler, faster, stronger. CVPR.
6. Thomas, H., et al. (2019). KPConv: Flexible and deformable convolution for point clouds. ICCV.
7. Hu, Q., et al. (2020). RandLA-Net: Efficient semantic segmentation of large-scale point clouds. CVPR.

### Active Learning for Point Clouds
8. Wu, T.-H., et al. (2021). ReDAL: Region-based and diversity-aware active learning for point cloud semantic segmentation. ICCV.
9. Hu, Z., et al. (2022). LiDAL: Inter-frame uncertainty based active learning for 3D LiDAR semantic segmentation. ECCV.

### Coastal LiDAR Applications
10. Young, A.P., et al. (2021). Comparing machine learning approaches for coastal cliff classification. Geomorphology.
11. Warrick, J.A., et al. (2023). LiDAR monitoring of coastal bluff erosion in Southern California. Journal of Coastal Research.

---

## Appendix B: Preliminary Code Structure

```
coastal_contrast/
├── configs/
│   ├── pretrain/
│   │   └── coastal_contrast.yaml
│   └── finetune/
│       ├── cliff_segmentation.yaml
│       └── change_detection.yaml
├── data/
│   ├── __init__.py
│   ├── las_dataset.py           # LAS file loading
│   ├── temporal_dataset.py      # Temporal pair sampling
│   ├── correspondence.py        # Temporal correspondence mining
│   └── transforms.py            # Point cloud augmentations
├── models/
│   ├── __init__.py
│   ├── ptv3_backbone.py         # PTv3 encoder (from repo)
│   ├── coastal_contrast.py      # Pre-training model
│   ├── seg_head.py              # Segmentation head
│   └── change_head.py           # Change detection head
├── losses/
│   ├── __init__.py
│   ├── temporal_contrast.py     # L_temporal
│   ├── spatial_contrast.py      # L_spatial
│   └── geometric.py             # L_geo
├── training/
│   ├── __init__.py
│   ├── pretrain.py              # Pre-training loop
│   ├── finetune.py              # Fine-tuning loop
│   └── active_learning.py       # Active learning utilities
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py               # mIoU, per-class IoU, etc.
│   └── visualization.py         # Point cloud visualization
├── scripts/
│   ├── preprocess_las.py        # Data preprocessing
│   ├── mine_correspondences.py  # Correspondence extraction
│   ├── train_pretrain.py        # Launch pre-training
│   ├── train_finetune.py        # Launch fine-tuning
│   └── inference.py             # Run inference on new data
└── tests/
    └── ...
```

---

## Appendix C: Annotation Guidelines (Draft)

### Class Definitions

**1. Beach (Class 0)**
- Bare sand, gravel, or cobble below cliff toe
- Includes wet sand at water line
- Excludes woody debris and wrack

**2. Cliff Face (Class 1)**
- Exposed geological material (sandstone, mudstone, conglomerate)
- Vertical to sub-vertical surfaces (>30° from horizontal)
- Includes failure scars and talus deposits

**3. Cliff Vegetation (Class 2)**
- Vegetation rooted on cliff face
- Includes ice plant, native coastal scrub
- Excludes vegetation overhanging from above

**4. Upland Vegetation (Class 3)**
- Vegetation on top of cliff or inland
- Trees, shrubs, grass
- Includes overhanging vegetation

**5. Riprap/Armor (Class 4)**
- Engineered rock or concrete protection
- Large angular boulders placed for erosion control
- Includes concrete rubble

**6. Seawall/Structure (Class 5)**
- Vertical concrete or wood structures
- Includes retaining walls, bulkheads
- Excludes natural cliff faces

**7. Water (Class 6)**
- Ocean water returns
- Includes wave foam and spray
- Note: Often filtered in preprocessing

**8. Infrastructure (Class 7)**
- Roads, sidewalks, buildings, fences
- Stairs, railings, viewing platforms
- Beach access structures

---

*Document Version 1.0 - January 2026*
*For internal use and grant applications*
