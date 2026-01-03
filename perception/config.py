"""Configuration for perception backend.

This module contains the configuration for the perception visibility checking system.
The oracle map defines which nodes have visible targets (for testing/simulation).
"""

from typing import Dict

# Default perception backend
DEFAULT_PERCEPTION_BACKEND = "node_oracle"

# Oracle map: node_id -> confidence (0.0 to 1.0)
# This simulates "ground truth" about where targets are actually visible.
# In production, this would be replaced by actual computer vision/perception.
NODE_ORACLE_MAP: Dict[int, float] = {}

# Relative pose oracle map: goal_key -> node_id -> {distance_m, bearing_rad, confidence?}
# Example:
#   NODE_ORACLE_RELPOSE_MAP = {
#       "find the red backpack": {
#           12: {"distance_m": 1.8, "bearing_rad": 0.2, "confidence": 0.9},
#           15: {"distance_m": 0.6, "bearing_rad": -0.1, "confidence": 1.0},
#       }
#   }
NODE_ORACLE_RELPOSE_MAP: Dict[str, Dict[int, Dict[str, float]]] = {}


def get_node_oracle_map() -> Dict[int, float]:
    """Get a copy of the node oracle map.
    
    Returns a copy to prevent accidental mutation of the global map.
    
    Returns:
        Dictionary mapping node_id to confidence (0.0 to 1.0)
    """
    return NODE_ORACLE_MAP.copy()


def normalize_goal_text(goal_text: str) -> str:
    """Normalize goal text for consistent lookup.
    
    Applies lowercase, strip, and collapses multiple whitespace to single space.
    
    Args:
        goal_text: Raw goal text string
        
    Returns:
        Normalized goal text for use as dictionary key
    """
    # TWEAK 4: Handle None/empty strings safely
    if not goal_text:
        return ""
    return " ".join(goal_text.lower().strip().split())


def get_node_oracle_relpose_map() -> Dict[str, Dict[int, Dict[str, float]]]:
    """Get a deep copy of the relpose oracle map to prevent accidental mutation.
    
    Returns a nested copy where both goal-level and node-level dicts are copied.
    
    Returns:
        Deep copy of NODE_ORACLE_RELPOSE_MAP
    """
    # FIX C: Deep copy for nested dicts
    return {
        goal_key: {
            node_id: data.copy()
            for node_id, data in nodes.items()
        }
        for goal_key, nodes in NODE_ORACLE_RELPOSE_MAP.items()
    }

