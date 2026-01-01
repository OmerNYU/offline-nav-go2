"""Visibility checking function for perception gate."""

import time
from typing import Any, Dict, List, Optional, Union

from perception.config import DEFAULT_PERCEPTION_BACKEND, get_node_oracle_map


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

