"""
GRIC: High-performance distance-based clustering and geometry engine.
"""

from .gric import Clusterer, version
from .image_cluster import ImageCluster

__all__ = ["Clusterer", "version", "ImageCluster"]
__version__ = "1.0.0"
