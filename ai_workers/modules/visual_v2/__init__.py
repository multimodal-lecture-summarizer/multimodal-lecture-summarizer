"""Visual modules package (v2)."""

from ai_workers.modules.visual_v2.scene_detector import SceneDetector
from ai_workers.modules.visual_v2.semantic import SemanticAnalyzer
from ai_workers.modules.visual_v2.object_tracker import ObjectDetectorTracker

__all__ = ["SceneDetector", "SemanticAnalyzer", "ObjectDetectorTracker"]
