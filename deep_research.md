# Deep Learning for Coastal LiDAR Segmentation: A Practical Guide

**Point Transformer V3 paired with self-supervised pre-training offers the most promising path forward** for classifying coastal geomorphological features from airborne LiDAR. This architecture achieves **73.4% mIoU on SemanticKITTI** and scales efficiently to million-point clouds—ideal for large-scale coastal monitoring. The combination of self-supervised pre-training (reducing labeled data requirements by 40-60%) and active learning (achieving 95% performance with <5% annotations) can dramatically reduce the labeling burden for novel coastal classes like cliff faces, riprap, and seawalls.

The core challenge is the domain gap: no large-scale benchmark exists for coastal geomorphology, so transfer learning from urban aerial datasets like **DALES** and **SensatUrban** provides the most practical starting point. RandLA-Net and KPConv remain the workhorses for aerial LiDAR processing, with proven performance on these datasets and efficient handling of point clouds exceeding one million points.

## The architecture landscape has shifted toward transformers

The progression from PointNet (2017) to Point Transformer V3 (2024) represents a **50+ percentage point improvement** in semantic segmentation accuracy on outdoor benchmarks. Three architectural families now dominate large-scale point cloud segmentation:

**Sparse convolution methods** like MinkowskiNet and SPVNAS voxelize point clouds and apply efficient sparse 3D convolutions. They achieve strong performance (66.4% mIoU on SemanticKITTI for SPVNAS) with fast inference, but voxelization loses fine geometric details critical for distinguishing coastal features like riprap from natural rock formations.

**Point-based methods** like RandLA-Net and KPConv operate directly on unstructured points. RandLA-Net processes **one million points in a single forward pass** at 22 FPS using random sampling and local feature aggregation—making it the efficiency champion for massive aerial surveys. KPConv's learnable kernel point convolutions achieve higher accuracy (58.8% on SemanticKITTI, **~77% on DALES**) at the cost of memory. Both have extensive aerial LiDAR validation.

**Transformer architectures** represent the current state-of-the-art. PTv3 introduced space-filling curve serialization (Hilbert/Z-order curves) to replace expensive k-NN queries, expanding the receptive field from 16 to **1024 points** while running **3.3× faster than PTv2** with 10× lower memory. It won the 2024 Waymo Challenge and achieves 83.4% mIoU on nuScenes. Critically, PTv3's design is sensor-agnostic—it makes no assumptions about cylindrical LiDAR geometry—making it more suitable for nadir-view aerial data than methods like Cylinder3D or SphereFormer designed for spinning automotive LiDAR.

For coastal airborne LiDAR specifically, avoid architectures designed for spinning LiDAR (Cylinder3D, SphereFormer) that assume radial point distributions. The recommended hierarchy is: **PTv3** for maximum accuracy when hardware permits, **KPConv** for balanced accuracy-efficiency, and **RandLA-Net** for processing massive datasets with limited resources.

## Self-supervised pre-training can halve your labeling requirements

Self-supervised learning for point clouds divides into two paradigms that address different needs. **Contrastive methods** like PointContrast and Contrastive Scene Contexts (CSC) learn by distinguishing between augmented views of the same scene. CSC demonstrates remarkable label efficiency: **0.1% point labels achieve 96% of full semantic segmentation performance**. However, these methods typically require multi-view overlapping scans with known correspondences—a constraint that may limit applicability for single-pass airborne surveys.

**Masked autoencoding methods** reconstruct masked portions of point clouds, similar to BERT's approach in NLP. Point-MAE and Point-BERT work well for object-level understanding but were designed for synthetic indoor data. For outdoor LiDAR, **GeoMAE** and **Occupancy-MAE** are specifically designed for large-scale sparse outdoor scenes. GeoMAE predicts multiple geometric targets (centroids, surface normals, curvature, occupancy) rather than just reconstructing coordinates—geometric features that are particularly informative for distinguishing natural terrain. Occupancy-MAE's range-aware masking accounts for the varying point density at different distances typical in LiDAR data.

The practical outcomes are substantial: Voxel-MAE achieves performance with **40% of the labeled data** that matches training from scratch. BEV-MAE with only 20% training data outperforms competitors using 100%. These methods have pre-trained weights available on GitHub for outdoor datasets like nuScenes and can be fine-tuned for coastal applications.

For a coastal LiDAR workflow, the recommended approach is: (1) Pre-train using DepthContrast (works with any single-view data without registration requirements) or GeoMAE (geometric features ideal for natural terrain), (2) Fine-tune with the limited coastal labels available, using CSC-style techniques if labels are extremely scarce.

## Transfer learning from urban aerial datasets offers a practical foundation

The most relevant pre-training datasets for coastal airborne LiDAR are:

- **DALES**: 10 km² of aerial LiDAR over Surrey, British Columbia at ~50 points/m² with 8 classes including ground, vegetation, and structures—the closest match to coastal survey geometry
- **SensatUrban**: 2.8 billion points over Birmingham and Cambridge with 13 urban classes, though photogrammetric rather than LiDAR
- **Semantic3D**: 4 billion terrestrial laser scanning points across 8 classes in outdoor scenes
- **SemanticKITTI/nuScenes**: Automotive LiDAR datasets that provide large-scale feature learning but with significant domain gap from aerial perspectives

Pre-trained weights are available for RandLA-Net (DALES, SemanticKITTI, S3DIS), KPConv (Semantic3D, Toronto3D), and PTv3 (multiple datasets via the Pointcept framework). MATLAB's Lidar Toolbox provides built-in RandLA-Net with DALES weights accessible via the `randlanet()` function—potentially the fastest path to a working baseline.

Fine-tuning strategy matters significantly. **Progressive unfreezing** works best: freeze the entire backbone, train only the segmentation head for 10-20 epochs, then unfreeze decoder layers for 20-30 epochs, and finally fine-tune the full network with a learning rate **10-100× lower** than initial training to prevent catastrophic forgetting. Layer-wise learning rates (higher for newly added coastal class heads, lower for pre-trained encoders) help balance adaptation with preservation of learned features.

The domain gap is real but surmountable. Studies show **30-50% mIoU degradation** when directly applying urban-trained models to new domains without adaptation, but proper fine-tuning can recover to within 5-15% of source domain performance. Domain adaptation techniques like CoSMix (compositional semantic mixing across domains) and Complete & Label (learning canonical representations independent of sensor characteristics) can potentially match or exceed supervised performance. A recent SAM-guided domain adaptation approach achieved ~20% mIoU improvement for nuScenes→SemanticKITTI transfer.

## Active learning makes annotation effort tractable

For coastal applications where each new class (riprap, seawalls, eroded cliff faces) requires manual annotation, active learning can reduce labeling effort by **5-20×**. The key insight is that not all points are equally informative—selecting uncertain or diverse samples yields far better models per annotation dollar than random labeling.

**Selection granularity** significantly impacts efficiency. Point-level selection offers fine-grained control but creates impractical annotation workflows. **Superpoint-based selection**—grouping geometrically coherent point clusters and selecting entire clusters for annotation—balances informativeness with practical annotation interfaces. VCCS (Voxel Cloud Connectivity Segmentation) in the Point Cloud Library provides standard supervoxel generation.

The most effective methods combine uncertainty and diversity. **ReDAL** (ICCV 2021) selects regions based on both entropy (uncertainty) and surface/color variation (informativeness), achieving significant improvements over random sampling. **LiDAL** (ECCV 2022) exploits temporal consistency in sequential scans—a well-trained model should predict consistently regardless of viewpoint, so high inter-frame divergence indicates samples worth labeling. With LiDAL, **<5% annotations achieve 95% of fully supervised performance** on SemanticKITTI and nuScenes.

Implementation is straightforward. Monte Carlo dropout (running inference with dropout active 10-50 times and measuring prediction variance) provides uncertainty estimates without architectural changes:

```python
def mc_dropout_uncertainty(model, data, n_samples=20):
    for m in model.modules():
        if isinstance(m, nn.Dropout): m.train()
    predictions = [F.softmax(model(data), dim=-1) for _ in range(n_samples)]
    entropy = -torch.sum(torch.stack(predictions).mean(0) * 
              torch.log(torch.stack(predictions).mean(0) + 1e-10), dim=-1)
    return entropy
```

Batch selection should add diversity via core-set selection (greedy k-center algorithm maximizing minimum distance to already-labeled points) applied after filtering by uncertainty. Start with **0.1-1% of points** labeled with stratified class coverage, then iterate: train → select uncertain/diverse regions → annotate → retrain.

Open-source implementations include ReDAL (github.com/tsunghan-wu/ReDAL), LiDAL (github.com/hzykent/LiDAL), and the Annotator framework (github.com/binhuixie/annotator) which handles cross-domain scenarios particularly well.

## Coastal geomorphology applications remain underexplored by deep learning

Direct application of modern point-based deep learning to coastal geomorphological classification is surprisingly sparse in the literature. Most existing coastal LiDAR work uses **traditional machine learning on rasterized DEMs** rather than operating directly on point clouds. A 2021 Geomorphology paper testing five ML algorithms for California cliff erosion found discriminant analysis (not deep learning) performed best—though this predates the transformer revolution.

The most relevant recent work combines LiDAR with other modalities. A December 2024 Nature Scientific Reports study achieved ~90% accuracy classifying coastal cliff degradation using Random Forest on combined bathymetric LiDAR and orthophotos in Google Earth Engine. Coastal dune habitat detection reached 78.6% accuracy using OBIA (Object-Based Image Analysis) with machine learning on UAV imagery combined with LiDAR-derived features.

For coastal vegetation specifically, deep learning shows clear advantages: U-Net achieved **83.8-85.3% overall accuracy** classifying 17 coastal land cover types versus 57-71% for Random Forest on the same data. Vegetation stratum classification using weakly-supervised deep learning on aerial LiDAR outperformed baselines by up to 30%.

This gap represents an opportunity. The architectures proven on urban aerial datasets (PTv3, KPConv, RandLA-Net) should transfer well to coastal scenes with appropriate fine-tuning. The key challenges are: (1) defining appropriate class taxonomies for coastal features, (2) handling the extreme class imbalance typical in coastal data (vegetation may dominate by orders of magnitude), and (3) addressing the domain gap between urban training data and natural coastal terrain.

## Practical implementation: Getting started with Pointcept

**Pointcept** (github.com/Pointcept/Pointcept) has emerged as the most comprehensive framework for point cloud perception, supporting PTv3, KPConv, RandLA-Net, MinkowskiNet, and many other architectures with unified training pipelines. It includes pre-trained weights for outdoor datasets and supports custom data formats.

For MinkowskiEngine-based sparse convolution (MinkUNet, SPVNAS), installation requires CUDA ≥10.1 and PyTorch ≥1.7:
```bash
pip install MinkowskiEngine
```

Data should be formatted as dictionaries containing coordinate arrays (N×3), optional features (intensity, RGB, return number), and semantic labels. Most frameworks expect point clouds normalized to a local coordinate frame with grid subsampling applied (0.02-0.05m for high-density aerial data).

- **Open3D** provides point cloud I/O, voxel downsampling, normal estimation, and RANSAC plane fitting
- **laspy** handles LAS/LAZ format standard for airborne LiDAR
- **torch-points-kernels** provides efficient KNN and ball query operations for custom architectures

For annotation, **labelCloud** (pip install labelCloud) offers free 3D labeling with bounding box and point selection. For larger projects, Segments.ai and Supervisely provide multi-frame fusion and AI-assisted segmentation that can dramatically accelerate labeling. AWS SageMaker Ground Truth offers cloud-based annotation with built-in quality control.

Visualization options include Open3D's interactive viewer, CloudCompare (GUI-based), Potree for web-based rendering of massive clouds, and TensorBoard integration for training monitoring.

## Recommended implementation pathway

For airborne coastal LiDAR classification with limited labeled data, a practical workflow would proceed as follows:

**Phase 1 (Week 1-2): Baseline establishment.** Start with RandLA-Net using DALES pre-trained weights via MATLAB's Lidar Toolbox or the PyTorch implementation at github.com/aRI0U/RandLA-Net-pytorch. This provides immediate results with minimal setup. Evaluate on a held-out coastal test set to establish baseline performance and identify which coastal classes (cliff, beach, riprap, vegetation, structures) prove most challenging.

**Phase 2 (Week 2-4): Self-supervised pre-training.** Using the full unlabeled coastal dataset, apply GeoMAE-style pre-training (github.com/Tsinghua-MARS-Lab/GeoMAE) to learn coastal-specific geometric features. Alternatively, if the data includes sequential scans with overlap, DepthContrast provides sensor-agnostic pre-training without registration requirements.

**Phase 3 (Week 4-6): Active learning loop.** Implement MC dropout uncertainty estimation and core-set diversity selection. Label the top 1-5% most informative regions identified by the active learner. Fine-tune the pre-trained model on this efficient labeled subset. Expect to reach 90-95% of fully supervised performance.

**Phase 4 (Week 6-8): Architecture upgrade.** If computational resources permit, migrate to PTv3 using the Pointcept framework for maximum accuracy. Transfer the feature representations learned during self-supervised pre-training.

The combination of modern transformer architectures, self-supervised pre-training on unlabeled coastal data, transfer learning from urban aerial datasets, and active learning for efficient annotation creates a powerful stack for tackling coastal geomorphological classification—even with the current absence of large-scale coastal point cloud benchmarks.

## Conclusion

The most impactful advances for coastal LiDAR segmentation are **PTv3's efficient scaling to large point clouds without LiDAR-specific assumptions**, **self-supervised methods like GeoMAE that can leverage unlabeled coastal data**, and **active learning frameworks achieving 95% performance with <5% annotations**. The critical gap is the lack of coastal-specific benchmarks—DALES provides the closest proxy for transfer learning, but building a coastal point cloud dataset with properly annotated geomorphological classes would significantly advance the field. The practical path forward combines pre-trained urban models, domain adaptation through self-supervised learning on unlabeled coastal data, and strategic active learning to minimize annotation costs for novel coastal classes.