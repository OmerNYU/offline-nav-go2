"""Tests for memory bridge belief state transitions."""

from runtime.memory_bridge import apply_memory_retrieval


def test_visible_never_demotes():
    """Test that visible status is never changed by retrieval."""
    belief = {
        "target_status": "visible",
        "goal_text": "test",
        "active_constraints": [],
        "candidate_nodes": [],
        "next_action": "approach",
        "current_node_id": None,
        "last_vlm_hypothesis": None,
        "rejection_reason": None
    }
    
    # Low score candidates
    candidates = [{"node_id": 1, "score": 0.1}]
    
    apply_memory_retrieval(belief, candidates, threshold=0.3)
    
    assert belief["target_status"] == "visible", "visible status should never change"
    assert belief["candidate_nodes"] == candidates, "candidate_nodes should be updated"


def test_promotion_searching_to_likely_in_memory():
    """Test promotion from searching to likely_in_memory with high score."""
    belief = {
        "target_status": "searching",
        "goal_text": "test",
        "active_constraints": [],
        "candidate_nodes": [],
        "next_action": "explore",
        "current_node_id": None,
        "last_vlm_hypothesis": None,
        "rejection_reason": None
    }
    
    # High score candidates
    candidates = [{"node_id": 2, "score": 0.8}]
    
    apply_memory_retrieval(belief, candidates, threshold=0.3)
    
    assert belief["target_status"] == "likely_in_memory", \
        "Should promote to likely_in_memory with high score"
    assert belief["candidate_nodes"] == candidates


def test_searching_stays_searching_with_low_score():
    """Test that searching remains searching with low score."""
    belief = {
        "target_status": "searching",
        "goal_text": "test",
        "active_constraints": [],
        "candidate_nodes": [],
        "next_action": "explore",
        "current_node_id": None,
        "last_vlm_hypothesis": None,
        "rejection_reason": None
    }
    
    # Low score candidates
    candidates = [{"node_id": 1, "score": 0.1}]
    
    apply_memory_retrieval(belief, candidates, threshold=0.3)
    
    assert belief["target_status"] == "searching", \
        "Should remain searching with low score"
    assert belief["candidate_nodes"] == candidates


def test_no_flapping_likely_in_memory():
    """Test that likely_in_memory does NOT demote to searching on weak retrieval."""
    belief = {
        "target_status": "likely_in_memory",
        "goal_text": "test",
        "active_constraints": [],
        "candidate_nodes": [{"node_id": 3, "score": 0.9}],
        "next_action": "goto_node",
        "current_node_id": None,
        "last_vlm_hypothesis": None,
        "rejection_reason": None
    }
    
    # Low score candidates (below threshold)
    candidates = [{"node_id": 1, "score": 0.1}]
    
    apply_memory_retrieval(belief, candidates, threshold=0.3)
    
    assert belief["target_status"] == "likely_in_memory", \
        "Should NOT demote from likely_in_memory to searching (no flapping)"
    assert belief["candidate_nodes"] == candidates, "candidate_nodes should still be updated"


def test_done_status_unchanged():
    """Test that done status is never changed by retrieval."""
    belief = {
        "target_status": "done",
        "goal_text": "test",
        "active_constraints": [],
        "candidate_nodes": [],
        "next_action": "stop",
        "current_node_id": 5,
        "last_vlm_hypothesis": None,
        "rejection_reason": None
    }
    
    # High score candidates
    candidates = [{"node_id": 2, "score": 0.9}]
    
    apply_memory_retrieval(belief, candidates, threshold=0.3)
    
    assert belief["target_status"] == "done", "done status should never change"
    assert belief["candidate_nodes"] == candidates


def test_unreachable_status_unchanged():
    """Test that unreachable status is never changed by retrieval."""
    belief = {
        "target_status": "unreachable",
        "goal_text": "test",
        "active_constraints": [],
        "candidate_nodes": [],
        "next_action": "explore",
        "current_node_id": None,
        "last_vlm_hypothesis": None,
        "rejection_reason": None
    }
    
    # High score candidates
    candidates = [{"node_id": 2, "score": 0.9}]
    
    apply_memory_retrieval(belief, candidates, threshold=0.3)
    
    assert belief["target_status"] == "unreachable", "unreachable status should never change"
    assert belief["candidate_nodes"] == candidates


def test_candidate_nodes_always_set():
    """Test that candidate_nodes is always updated for all statuses."""
    statuses = ["visible", "searching", "likely_in_memory", "done", "unreachable"]
    
    for status in statuses:
        belief = {
            "target_status": status,
            "goal_text": "test",
            "active_constraints": [],
            "candidate_nodes": [],
            "next_action": "explore",
            "current_node_id": None,
            "last_vlm_hypothesis": None,
            "rejection_reason": None
        }
        
        candidates = [{"node_id": 1, "score": 0.5}]
        apply_memory_retrieval(belief, candidates, threshold=0.3)
        
        assert belief["candidate_nodes"] == candidates, \
            f"candidate_nodes should be set for status={status}"


def test_empty_candidates():
    """Test handling of empty candidates list."""
    belief = {
        "target_status": "searching",
        "goal_text": "test",
        "active_constraints": [],
        "candidate_nodes": [{"node_id": 1, "score": 0.5}],
        "next_action": "explore",
        "current_node_id": None,
        "last_vlm_hypothesis": None,
        "rejection_reason": None
    }
    
    # Empty candidates
    candidates = []
    apply_memory_retrieval(belief, candidates, threshold=0.3)
    
    assert belief["candidate_nodes"] == [], "candidate_nodes should be empty list"
    assert belief["target_status"] == "searching", "Should remain searching with no candidates"


def test_threshold_boundary():
    """Test behavior at threshold boundary."""
    # Test exactly at threshold
    belief = {
        "target_status": "searching",
        "goal_text": "test",
        "active_constraints": [],
        "candidate_nodes": [],
        "next_action": "explore",
        "current_node_id": None,
        "last_vlm_hypothesis": None,
        "rejection_reason": None
    }
    
    # Exactly at threshold
    candidates = [{"node_id": 1, "score": 0.3}]
    apply_memory_retrieval(belief, candidates, threshold=0.3)
    
    assert belief["target_status"] == "likely_in_memory", \
        "Should promote when score equals threshold"
    
    # Just below threshold
    belief["target_status"] = "searching"
    candidates = [{"node_id": 1, "score": 0.299}]
    apply_memory_retrieval(belief, candidates, threshold=0.3)
    
    assert belief["target_status"] == "searching", \
        "Should not promote when score is below threshold"

