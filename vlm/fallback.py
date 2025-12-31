"""Safe fallback hypothesis generation.

Provides deterministic fallback when VLM generation fails or is unavailable.
"""

from typing import Any, Dict, List


def generate_fallback_hypothesis(
    belief: Dict[str, Any],
    candidates: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Generate a safe, schema-valid fallback hypothesis.
    
    Deterministic logic (no heuristics):
    - If target_status == "done": stop
    - Elif target_status == "visible": approach
    - Elif candidates non-empty: goto_node (top candidate)
    - Else: explore
    
    Args:
        belief: Current belief state dict
        candidates: List of candidate nodes with {node_id, score}
    
    Returns:
        Schema-valid VLMHypothesis dict
    """
    target_status_belief = belief.get("target_status", "searching")
    
    # Case 1: Goal completed
    if target_status_belief == "done":
        return {
            "target_status": "visible",
            "action": "stop",
            "confidence": 0.9,
            "rationale": "Goal completed"
        }
    
    # Case 2: Target is visible
    if target_status_belief == "visible":
        return {
            "target_status": "visible",
            "action": "approach",
            "confidence": 0.7,
            "rationale": "Approaching visible target",
            "navigation_goal": {
                "type": "pose_relative",
                "distance_meters": 1.0,
                "angle_degrees": 0.0,
                "standoff_distance": 0.5
            }
        }
    
    # Case 3: Candidates available from memory
    if candidates:
        top_candidate_id = candidates[0]["node_id"]
        return {
            "target_status": "not_visible",
            "action": "goto_node",
            "confidence": 0.5,
            "rationale": "Navigating to candidate memory node",
            "navigation_goal": {
                "type": "node_id",
                "node_id": top_candidate_id
            }
        }
    
    # Case 4: No information available, explore
    return {
        "target_status": "not_visible",
        "action": "explore",
        "confidence": 0.3,
        "rationale": "Exploring for target"
    }

