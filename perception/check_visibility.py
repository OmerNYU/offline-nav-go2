"""Visibility checking function for perception gate."""

import time
from typing import Any, Dict, List, Optional, Union

from perception.config import (
    DEFAULT_PERCEPTION_BACKEND,
    get_node_oracle_map,
    normalize_goal_text,
    get_node_oracle_relpose_map
)


def check_visibility(
    goal_text: str,
    current_node_id: Optional[int],
    memory_context: Optional[List],
    belief_state: Union[object, Dict[str, Any]],
    backend: Optional[str] = None,
    config: Optional[Dict] = None
) -> Dict[str, Any]:
    """Check if the goal is visible at the current location.
    
    This is the ONLY authority that can determine visibility.
    VLM output is NEVER trusted for visibility transitions.
    
    Args:
        goal_text: The goal description (e.g., "red backpack")
        current_node_id: Current node ID, or None if not at a node
        memory_context: Memory context list (may be None)
        belief_state: Current belief state (dict or object)
        backend: Perception backend to use, defaults to DEFAULT_PERCEPTION_BACKEND
        config: Optional config override dict (for testing)
    
    Returns:
        VisibilityResult dict with keys:
            - is_visible: bool indicating if target is visible
            - confidence: float 0.0 to 1.0
            - backend: str indicating backend used
            - latency_ms: int milliseconds taken
            - distance_m: Optional[float] distance to target in meters (relpose only)
            - bearing_rad: Optional[float] bearing to target in radians (relpose only)
            - target_goal_key: Optional[str] normalized goal text (relpose only, debug)
            - evidence: dict with:
                - reason: str explanation
                - node_id: int or None
                - extra: dict with additional info
    """
    start_time = time.perf_counter()
    
    # Resolve backend
    if backend is None:
        backend = DEFAULT_PERCEPTION_BACKEND
    
    # Initialize result structure
    result: Dict[str, Any] = {
        "is_visible": False,
        "confidence": 0.0,
        "backend": backend,
        "latency_ms": 0,
        "distance_m": None,      # NEW: initialize to None
        "bearing_rad": None,     # NEW: initialize to None
        "target_goal_key": None, # NEW: initialize to None
        "evidence": {
            "reason": "not_checked",
            "node_id": current_node_id,
            "extra": {}
        }
    }
    
    try:
        if backend == "node_oracle":
            # Get oracle map from config or default
            if config and "oracle_map" in config:
                oracle_map = config["oracle_map"]
            else:
                oracle_map = get_node_oracle_map()
            
            # Check if current node is in oracle map
            if current_node_id is not None and current_node_id in oracle_map:
                confidence = oracle_map[current_node_id]
                result["is_visible"] = True
                result["confidence"] = confidence
                result["evidence"]["reason"] = "oracle_hit"
                result["evidence"]["extra"]["oracle_confidence"] = confidence
            else:
                result["is_visible"] = False
                result["confidence"] = 0.0
                result["evidence"]["reason"] = "oracle_miss"
                if current_node_id is None:
                    result["evidence"]["extra"]["note"] = "current_node_id is None"
                else:
                    result["evidence"]["extra"]["note"] = f"node {current_node_id} not in oracle map"
        elif backend == "node_oracle_relpose":
            # Get relpose map from config or default
            if config and "relpose_map" in config:
                relpose_map = config["relpose_map"]
            else:
                relpose_map = get_node_oracle_relpose_map()
            
            # Normalize goal for consistent lookup
            goal_key = normalize_goal_text(goal_text)
            result["target_goal_key"] = goal_key
            
            # Lookup (goal_key, current_node_id)
            if current_node_id is not None and goal_key in relpose_map:
                node_data = relpose_map[goal_key].get(current_node_id)
                
                if node_data:
                    # Extract fields (use .get() for safety)
                    distance_m = node_data.get("distance_m")
                    bearing_rad = node_data.get("bearing_rad")
                    confidence = node_data.get("confidence")
                    
                    # Fallback confidence: try visibility map, then default to 1.0
                    if confidence is None:
                        oracle_map = get_node_oracle_map()
                        confidence = oracle_map.get(current_node_id, 1.0)
                    
                    # FIX D: Clamp confidence to [0.0, 1.0]
                    try:
                        confidence = float(confidence)
                        confidence = max(0.0, min(1.0, confidence))
                    except (TypeError, ValueError):
                        confidence = 0.0  # Invalid value → safe default
                    
                    # Visibility decision: confidence > 0.0 (deterministic)
                    result["is_visible"] = confidence > 0.0
                    result["confidence"] = confidence
                    result["distance_m"] = distance_m
                    result["bearing_rad"] = bearing_rad
                    result["evidence"]["reason"] = "relpose_hit"
                    result["evidence"]["extra"]["goal_key"] = goal_key
                    result["evidence"]["extra"]["distance_m"] = distance_m
                    result["evidence"]["extra"]["bearing_rad"] = bearing_rad
                    result["evidence"]["extra"]["confidence"] = confidence
                else:
                    # Node not mapped for this goal
                    result["is_visible"] = False
                    result["confidence"] = 0.0
                    # TWEAK 3: Explicitly set to None in miss paths
                    result["distance_m"] = None
                    result["bearing_rad"] = None
                    result["evidence"]["reason"] = "relpose_miss"
                    result["evidence"]["extra"]["note"] = f"node {current_node_id} not mapped for goal '{goal_key}'"
            else:
                # Goal or node not found
                result["is_visible"] = False
                result["confidence"] = 0.0
                # TWEAK 3: Explicitly set to None in miss paths
                result["distance_m"] = None
                result["bearing_rad"] = None
                result["evidence"]["reason"] = "relpose_miss"
                if current_node_id is None:
                    result["evidence"]["extra"]["note"] = "current_node_id is None"
                elif goal_key not in relpose_map:
                    result["evidence"]["extra"]["note"] = f"goal_key '{goal_key}' not in relpose map"
                else:
                    result["evidence"]["extra"]["note"] = f"node {current_node_id} not in goal '{goal_key}' mapping"
        else:
            # Unknown backend
            result["evidence"]["reason"] = "unknown_backend"
            result["evidence"]["extra"]["error"] = f"Unknown backend: {backend}"
    
    except Exception as e:
        # Fail-safe: return safe default on any exception
        result["is_visible"] = False
        result["confidence"] = 0.0
        result["evidence"]["reason"] = "exception"
        result["evidence"]["extra"]["error"] = str(e)
        result["evidence"]["extra"]["exception_type"] = type(e).__name__
    
    # Calculate latency
    end_time = time.perf_counter()
    result["latency_ms"] = int((end_time - start_time) * 1000)
    
    return result

