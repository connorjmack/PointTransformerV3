"""
Data processing modules for CoastalContrast.

Includes:
- catalog: Survey indexing and metadata management
- registration: ICP co-registration
- correspondence: Temporal correspondence mining
- preprocess: Enhanced preprocessing with normals and multi-resolution
- transforms: Point cloud augmentations
- temporal_dataset: DataLoader for temporal pairs
"""

from .catalog import SurveyCatalog, SurveyMetadata

__all__ = ["SurveyCatalog", "SurveyMetadata"]
