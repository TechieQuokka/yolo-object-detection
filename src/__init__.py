"""YOLO Object Detection System"""

__version__ = "1.0.0"
__author__ = "YOLO Detection Team"

from .config import YOLOConfig
from .model.detector import YOLODetector
from .data.loader import DataLoader
from .output.visualizer import Visualizer

__all__ = ["YOLOConfig", "YOLODetector", "DataLoader", "Visualizer"]
