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


def get_node_oracle_map() -> Dict[int, float]:
    """Get a copy of the node oracle map.
    
    Returns a copy to prevent accidental mutation of the global map.
    
    Returns:
        Dictionary mapping node_id to confidence (0.0 to 1.0)
    """
    return NODE_ORACLE_MAP.copy()

