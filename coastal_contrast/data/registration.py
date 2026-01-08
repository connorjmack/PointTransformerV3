"""
Co-Registration Module for Temporal LiDAR Surveys

Aligns multiple LiDAR surveys to a common reference epoch using ICP
(Iterative Closest Point) registration. Essential for temporal
correspondence mining in contrastive learning.

Usage:
    from coastal_contrast.data.registration import SurveyRegistrar

    registrar = SurveyRegistrar(reference_path="/path/to/reference.las")
    transform, rmse = registrar.register_survey("/path/to/source.las")
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any
from dataclasses import dataclass
import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RegistrationResult:
    """Result of point cloud registration."""
    transformation: np.ndarray      # 4x4 transformation matrix
    rmse: float                     # Root mean square error
    fitness: float                  # Overlap ratio (0-1)
    correspondence_set_size: int   # Number of corresponding points
    converged: bool                # Whether ICP converged
    source_path: str               # Path to source file
    reference_path: str            # Path to reference file


def load_point_cloud_from_las(
    las_path: str,
    max_points: Optional[int] = None,
    voxel_size: Optional[float] = None
) -> 'o3d.geometry.PointCloud':
    """
    Load point cloud from LAS file into Open3D format.

    Args:
        las_path: Path to LAS/LAZ file
        max_points: Maximum points to load (random subsample if exceeded)
        voxel_size: If provided, voxel downsample after loading

    Returns:
        Open3D PointCloud object
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d required. Install with: pip install open3d")
    if not HAS_LASPY:
        raise ImportError("laspy required. Install with: pip install laspy")

    logger.info(f"Loading {las_path}")
    las = laspy.read(las_path)

    # Extract coordinates
    points = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)

    # Subsample if too many points
    if max_points and len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        logger.info(f"Subsampled to {max_points} points")

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Voxel downsample if requested
    if voxel_size:
        pcd = pcd.voxel_down_sample(voxel_size)
        logger.info(f"Voxel downsampled to {len(pcd.points)} points")

    return pcd


def estimate_normals(
    pcd: 'o3d.geometry.PointCloud',
    radius: float = 1.0,
    max_nn: int = 30
) -> 'o3d.geometry.PointCloud':
    """
    Estimate surface normals for point cloud.

    Args:
        pcd: Open3D PointCloud
        radius: Search radius for normal estimation
        max_nn: Maximum neighbors to consider

    Returns:
        PointCloud with normals
    """
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=max_nn
        )
    )
    return pcd


def extract_stable_features(
    pcd: 'o3d.geometry.PointCloud',
    voxel_size: float = 0.5,
    planarity_threshold: float = 0.8
) -> Tuple['o3d.geometry.PointCloud', np.ndarray]:
    """
    Extract stable features (planar surfaces) for robust registration.

    Stable features are typically:
    - Roads, sidewalks (horizontal planes)
    - Building walls (vertical planes)
    - Bedrock outcrops

    Args:
        pcd: Input point cloud with normals
        voxel_size: Voxel size for feature extraction
        planarity_threshold: Minimum planarity score (0-1)

    Returns:
        Tuple of (filtered point cloud, planarity scores)
    """
    if not pcd.has_normals():
        estimate_normals(pcd)

    # Compute planarity using PCA on local neighborhoods
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    planarity_scores = []

    for i in range(len(points)):
        # Find neighbors
        [_, idx, _] = pcd_tree.search_radius_vector_3d(points[i], voxel_size * 2)

        if len(idx) < 10:
            planarity_scores.append(0.0)
            continue

        # Get neighbor points
        neighbors = points[idx]

        # PCA to get eigenvalues
        centered = neighbors - neighbors.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]

        # Planarity = (λ2 - λ3) / λ1
        if eigenvalues[0] > 1e-6:
            planarity = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0]
        else:
            planarity = 0.0

        planarity_scores.append(planarity)

    planarity_scores = np.array(planarity_scores)

    # Filter by planarity
    mask = planarity_scores > planarity_threshold
    indices = np.where(mask)[0]

    filtered_pcd = pcd.select_by_index(indices)
    logger.info(f"Extracted {len(filtered_pcd.points)} stable features "
                f"({100*len(indices)/len(points):.1f}% of points)")

    return filtered_pcd, planarity_scores


def compute_fpfh_features(
    pcd: 'o3d.geometry.PointCloud',
    voxel_size: float = 0.5
) -> 'o3d.pipelines.registration.Feature':
    """
    Compute FPFH (Fast Point Feature Histogram) features for global registration.

    Args:
        pcd: Point cloud with normals
        voxel_size: Voxel size (affects feature radius)

    Returns:
        FPFH features
    """
    if not pcd.has_normals():
        estimate_normals(pcd, radius=voxel_size * 2)

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5,
            max_nn=100
        )
    )
    return fpfh


def global_registration_ransac(
    source: 'o3d.geometry.PointCloud',
    target: 'o3d.geometry.PointCloud',
    source_fpfh: 'o3d.pipelines.registration.Feature',
    target_fpfh: 'o3d.pipelines.registration.Feature',
    voxel_size: float = 0.5,
    distance_threshold: Optional[float] = None
) -> 'o3d.pipelines.registration.RegistrationResult':
    """
    Global registration using RANSAC with FPFH features.

    Use this for initial alignment when point clouds may be far apart.

    Args:
        source: Source point cloud
        target: Target point cloud
        source_fpfh: Source FPFH features
        target_fpfh: Target FPFH features
        voxel_size: Voxel size used for feature computation
        distance_threshold: Maximum correspondence distance

    Returns:
        Registration result with transformation
    """
    if distance_threshold is None:
        distance_threshold = voxel_size * 1.5

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )

    return result


def icp_point_to_plane(
    source: 'o3d.geometry.PointCloud',
    target: 'o3d.geometry.PointCloud',
    initial_transform: Optional[np.ndarray] = None,
    max_correspondence_distance: float = 0.5,
    max_iterations: int = 100,
    relative_fitness: float = 1e-6,
    relative_rmse: float = 1e-6
) -> 'o3d.pipelines.registration.RegistrationResult':
    """
    Fine registration using point-to-plane ICP.

    Point-to-plane ICP converges faster and is more accurate than
    point-to-point ICP for planar surfaces common in built environments.

    Args:
        source: Source point cloud (will be transformed)
        target: Target point cloud (reference)
        initial_transform: Initial 4x4 transformation (identity if None)
        max_correspondence_distance: Maximum distance for correspondences
        max_iterations: Maximum ICP iterations
        relative_fitness: Convergence threshold for fitness
        relative_rmse: Convergence threshold for RMSE

    Returns:
        Registration result with transformation
    """
    if initial_transform is None:
        initial_transform = np.eye(4)

    # Ensure normals exist
    if not source.has_normals():
        estimate_normals(source)
    if not target.has_normals():
        estimate_normals(target)

    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance,
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=relative_fitness,
            relative_rmse=relative_rmse,
            max_iteration=max_iterations
        )
    )

    return result


def icp_point_to_point(
    source: 'o3d.geometry.PointCloud',
    target: 'o3d.geometry.PointCloud',
    initial_transform: Optional[np.ndarray] = None,
    max_correspondence_distance: float = 0.5,
    max_iterations: int = 100
) -> 'o3d.pipelines.registration.RegistrationResult':
    """
    Fine registration using point-to-point ICP.

    Simpler than point-to-plane, doesn't require normals.

    Args:
        source: Source point cloud
        target: Target point cloud
        initial_transform: Initial transformation
        max_correspondence_distance: Maximum distance for correspondences
        max_iterations: Maximum iterations

    Returns:
        Registration result
    """
    if initial_transform is None:
        initial_transform = np.eye(4)

    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance,
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )

    return result


class SurveyRegistrar:
    """
    Register multiple LiDAR surveys to a common reference epoch.

    This class handles the full registration pipeline:
    1. Load and preprocess point clouds
    2. Optional: Extract stable features for robust registration
    3. Optional: Global registration (RANSAC) for initial alignment
    4. Fine registration (ICP) for precise alignment
    5. Save transformation matrices

    Example:
        registrar = SurveyRegistrar(
            reference_path="/data/reference_survey.las",
            voxel_size=0.1,
            use_stable_features=True
        )

        # Register a single survey
        result = registrar.register("/data/survey_2020_01_05.las")
        print(f"RMSE: {result.rmse:.4f}")

        # Apply transformation to points
        transformed = registrar.transform_points(points, result.transformation)
    """

    def __init__(
        self,
        reference_path: str,
        voxel_size: float = 0.1,
        max_points: int = 1000000,
        use_stable_features: bool = False,
        use_global_registration: bool = False,
        icp_max_distance: float = 0.5,
        icp_max_iterations: int = 100
    ):
        """
        Initialize registrar with reference point cloud.

        Args:
            reference_path: Path to reference LAS file
            voxel_size: Voxel size for downsampling
            max_points: Maximum points to use (for memory)
            use_stable_features: Extract stable features for registration
            use_global_registration: Use RANSAC for initial alignment
            icp_max_distance: Maximum correspondence distance for ICP
            icp_max_iterations: Maximum ICP iterations
        """
        if not HAS_OPEN3D:
            raise ImportError("open3d required. Install with: pip install open3d")

        self.reference_path = reference_path
        self.voxel_size = voxel_size
        self.max_points = max_points
        self.use_stable_features = use_stable_features
        self.use_global_registration = use_global_registration
        self.icp_max_distance = icp_max_distance
        self.icp_max_iterations = icp_max_iterations

        # Load reference point cloud
        logger.info(f"Loading reference: {reference_path}")
        self.reference_pcd = load_point_cloud_from_las(
            reference_path,
            max_points=max_points,
            voxel_size=voxel_size
        )

        # Estimate normals for point-to-plane ICP
        estimate_normals(self.reference_pcd, radius=voxel_size * 2)

        # Extract stable features if requested
        if use_stable_features:
            self.reference_stable, _ = extract_stable_features(
                self.reference_pcd, voxel_size=voxel_size
            )
        else:
            self.reference_stable = self.reference_pcd

        # Compute FPFH if global registration needed
        if use_global_registration:
            self.reference_fpfh = compute_fpfh_features(
                self.reference_stable, voxel_size=voxel_size
            )
        else:
            self.reference_fpfh = None

        logger.info(f"Reference loaded: {len(self.reference_pcd.points)} points")

    def register(
        self,
        source_path: str,
        initial_transform: Optional[np.ndarray] = None
    ) -> RegistrationResult:
        """
        Register a source survey to the reference.

        Args:
            source_path: Path to source LAS file
            initial_transform: Optional initial transformation

        Returns:
            RegistrationResult with transformation and metrics
        """
        # Load source
        source_pcd = load_point_cloud_from_las(
            source_path,
            max_points=self.max_points,
            voxel_size=self.voxel_size
        )
        estimate_normals(source_pcd, radius=self.voxel_size * 2)

        # Extract stable features if configured
        if self.use_stable_features:
            source_stable, _ = extract_stable_features(
                source_pcd, voxel_size=self.voxel_size
            )
        else:
            source_stable = source_pcd

        # Global registration if configured
        if self.use_global_registration and initial_transform is None:
            logger.info("Running global registration (RANSAC)...")
            source_fpfh = compute_fpfh_features(source_stable, self.voxel_size)
            global_result = global_registration_ransac(
                source_stable, self.reference_stable,
                source_fpfh, self.reference_fpfh,
                voxel_size=self.voxel_size
            )
            initial_transform = global_result.transformation
            logger.info(f"Global registration fitness: {global_result.fitness:.4f}")

        # Fine registration with ICP
        logger.info("Running ICP registration...")
        icp_result = icp_point_to_plane(
            source_stable, self.reference_stable,
            initial_transform=initial_transform,
            max_correspondence_distance=self.icp_max_distance,
            max_iterations=self.icp_max_iterations
        )

        # Determine if converged
        converged = icp_result.fitness > 0.5 and icp_result.inlier_rmse < self.icp_max_distance

        result = RegistrationResult(
            transformation=icp_result.transformation,
            rmse=icp_result.inlier_rmse,
            fitness=icp_result.fitness,
            correspondence_set_size=len(icp_result.correspondence_set),
            converged=converged,
            source_path=source_path,
            reference_path=self.reference_path
        )

        logger.info(f"Registration complete - RMSE: {result.rmse:.4f}, "
                   f"Fitness: {result.fitness:.4f}, Converged: {converged}")

        return result

    @staticmethod
    def transform_points(
        points: np.ndarray,
        transformation: np.ndarray
    ) -> np.ndarray:
        """
        Apply transformation to points.

        Args:
            points: (N, 3) array of points
            transformation: 4x4 transformation matrix

        Returns:
            Transformed points (N, 3)
        """
        # Convert to homogeneous coordinates
        ones = np.ones((points.shape[0], 1))
        points_h = np.hstack([points, ones])

        # Apply transformation
        transformed_h = (transformation @ points_h.T).T

        # Convert back
        return transformed_h[:, :3]

    @staticmethod
    def save_transformation(
        transformation: np.ndarray,
        output_path: str,
        metadata: Optional[Dict] = None
    ):
        """Save transformation matrix to file."""
        data = {
            'transformation': transformation.tolist(),
            'metadata': metadata or {}
        }

        import json
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_transformation(input_path: str) -> Tuple[np.ndarray, Dict]:
        """Load transformation matrix from file."""
        import json
        with open(input_path, 'r') as f:
            data = json.load(f)

        transformation = np.array(data['transformation'])
        metadata = data.get('metadata', {})

        return transformation, metadata


def apply_transformation_to_las(
    input_path: str,
    output_path: str,
    transformation: np.ndarray
) -> None:
    """
    Apply transformation to LAS file and save result.

    Args:
        input_path: Input LAS file
        output_path: Output LAS file
        transformation: 4x4 transformation matrix
    """
    if not HAS_LASPY:
        raise ImportError("laspy required")

    logger.info(f"Transforming {input_path}")

    # Read original file
    las = laspy.read(input_path)

    # Get points
    points = np.vstack((las.x, las.y, las.z)).T

    # Transform
    transformed = SurveyRegistrar.transform_points(points, transformation)

    # Update coordinates
    las.x = transformed[:, 0]
    las.y = transformed[:, 1]
    las.z = transformed[:, 2]

    # Save
    las.write(output_path)
    logger.info(f"Saved to {output_path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Register LiDAR surveys")
    parser.add_argument("--reference", "-r", required=True, help="Reference LAS file")
    parser.add_argument("--source", "-s", required=True, help="Source LAS file to register")
    parser.add_argument("--output", "-o", help="Output path for transformed LAS")
    parser.add_argument("--voxel-size", type=float, default=0.1, help="Voxel size")
    parser.add_argument("--max-points", type=int, default=1000000, help="Max points")
    parser.add_argument("--use-global", action="store_true", help="Use global registration")

    args = parser.parse_args()

    registrar = SurveyRegistrar(
        reference_path=args.reference,
        voxel_size=args.voxel_size,
        max_points=args.max_points,
        use_global_registration=args.use_global
    )

    result = registrar.register(args.source)

    print(f"\nRegistration Result:")
    print(f"  RMSE: {result.rmse:.4f}")
    print(f"  Fitness: {result.fitness:.4f}")
    print(f"  Converged: {result.converged}")

    if args.output:
        apply_transformation_to_las(args.source, args.output, result.transformation)
