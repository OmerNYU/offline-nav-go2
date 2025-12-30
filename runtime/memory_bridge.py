"""Bridge between semantic memory retrieval and belief state updates."""

from typing import Any, Dict


def apply_memory_retrieval(
    belief: Dict[str, Any],
    candidates: list[dict],
    threshold: float = 0.3
) -> Dict[str, Any]:
    """Apply memory retrieval results to belief state.
    
    Updates belief["candidate_nodes"] and potentially belief["target_status"]
    based on retrieval results, respecting monotonicity constraints to prevent
    status flapping.
    
    Status transition rules:
    - "visible": NEVER changed (only candidate_nodes updated)
    - "done" / "unreachable": NEVER changed (only candidate_nodes updated)
    - "likely_in_memory": NOT demoted to "searching" (no flapping)
    - "searching": Promoted to "likely_in_memory" if best_score >= threshold
    
    Args:
        belief: Current belief state dictionary (modified in-place).
        candidates: List of retrieved candidates with node_id and score.
        threshold: Score threshold for promotion to "likely_in_memory".
        
    Returns:
        The modified belief dictionary (same object as input).
    """
    # Always update candidate_nodes (even if empty) for schema safety
    belief["candidate_nodes"] = candidates
    
    # Get current status
    current_status = belief.get("target_status", "searching")
    
    # Apply status transition rules
    if current_status == "visible":
        # NEVER change visible status due to retrieval
        pass
    elif current_status in ["done", "unreachable"]:
        # NEVER change terminal states due to retrieval
        pass
    elif current_status == "likely_in_memory":
        # Do NOT demote to searching (no flapping)
        # Keep status as-is
        pass
    elif current_status == "searching":
        # Promote to likely_in_memory if we have strong candidates
        if candidates and candidates[0]["score"] >= threshold:
            belief["target_status"] = "likely_in_memory"
        # Otherwise remain searching
    # else: keep current status for any other states
    
    return belief

