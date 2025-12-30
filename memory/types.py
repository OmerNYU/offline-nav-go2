"""Core types for semantic memory representation."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Pose2D:
    """2D pose representation (x, y, yaw).
    
    Attributes:
        x: X coordinate in meters.
        y: Y coordinate in meters.
        yaw: Orientation in radians.
    """
    x: float
    y: float
    yaw: float


@dataclass
class MemoryNode:
    """Semantic memory node representing a location with metadata.
    
    Attributes:
        node_id: Unique identifier for the node (non-negative integer).
        pose: 2D pose of the node.
        embedding: Optional pre-computed embedding vector.
        tags: List of semantic tags (e.g., ["kitchen", "doorway"]).
        summary: Natural language description of the node.
    """
    node_id: int
    pose: Pose2D
    embedding: Optional[list[float]]
    tags: list[str]
    summary: str

