"""Tests for fallback hypothesis generation."""

from runtime.schema_loader import load_hypothesis_validator, validate_or_error
from vlm.fallback import generate_fallback_hypothesis


def test_fallback_when_done():
    """Test fallback when target_status is 'done'."""
    belief = {
        "target_status": "done",
        "goal_text": "test",
        "active_constraints": [],
        "candidate_nodes": [],
        "next_action": "stop",
        "current_node_id": None,
        "last_vlm_hypothesis": None,
        "rejection_reason": None
    }
    candidates = []
    
    hypothesis = generate_fallback_hypothesis(belief, candidates)
    
    assert hypothesis["action"] == "stop"
    assert hypothesis["target_status"] == "visible"
    assert hypothesis["confidence"] == 0.9
    assert "rationale" in hypothesis


def test_fallback_when_visible():
    """Test fallback when target_status is 'visible'."""
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
    candidates = []
    
    hypothesis = generate_fallback_hypothesis(belief, candidates)
    
    assert hypothesis["action"] == "approach"
    assert hypothesis["target_status"] == "visible"
    assert "navigation_goal" in hypothesis
    assert hypothesis["navigation_goal"]["type"] == "pose_relative"
    assert "distance_meters" in hypothesis["navigation_goal"]
    assert "angle_degrees" in hypothesis["navigation_goal"]
    assert "standoff_distance" in hypothesis["navigation_goal"]


def test_fallback_with_candidates():
    """Test fallback with candidate nodes."""
    belief = {
        "target_status": "searching",
        "goal_text": "test",
        "active_constraints": [],
        "candidate_nodes": [{"node_id": 5, "score": 0.8}],
        "next_action": "explore",
        "current_node_id": None,
        "last_vlm_hypothesis": None,
        "rejection_reason": None
    }
    candidates = [{"node_id": 5, "score": 0.8}, {"node_id": 3, "score": 0.6}]
    
    hypothesis = generate_fallback_hypothesis(belief, candidates)
    
    assert hypothesis["action"] == "goto_node"
    assert hypothesis["target_status"] == "not_visible"
    assert "navigation_goal" in hypothesis
    assert hypothesis["navigation_goal"]["type"] == "node_id"
    assert hypothesis["navigation_goal"]["node_id"] == 5  # top candidate


def test_fallback_without_candidates():
    """Test fallback without candidate nodes."""
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
    candidates = []
    
    hypothesis = generate_fallback_hypothesis(belief, candidates)
    
    assert hypothesis["action"] == "explore"
    assert hypothesis["target_status"] == "not_visible"
    assert hypothesis["confidence"] == 0.3


def test_fallback_outputs_are_schema_valid():
    """Test that all fallback outputs are schema-valid."""
    validator = load_hypothesis_validator()
    
    test_cases = [
        ({"target_status": "done"}, []),
        ({"target_status": "visible"}, []),
        ({"target_status": "searching"}, [{"node_id": 5, "score": 0.8}]),
        ({"target_status": "searching"}, []),
        ({"target_status": "likely_in_memory"}, [{"node_id": 3, "score": 0.7}]),
    ]
    
    for belief_partial, candidates in test_cases:
        belief = {
            "target_status": belief_partial.get("target_status", "searching"),
            "goal_text": "test",
            "active_constraints": [],
            "candidate_nodes": candidates,
            "next_action": "explore",
            "current_node_id": None,
            "last_vlm_hypothesis": None,
            "rejection_reason": None
        }
        
        hypothesis = generate_fallback_hypothesis(belief, candidates)
        
        is_valid, error = validate_or_error(validator, hypothesis)
        assert is_valid, f"Fallback hypothesis invalid for {belief_partial}: {error}"


def test_fallback_deterministic():
    """Test that fallback is deterministic (same inputs → same outputs)."""
    belief = {
        "target_status": "searching",
        "goal_text": "test",
        "active_constraints": [],
        "candidate_nodes": [{"node_id": 5, "score": 0.8}],
        "next_action": "explore",
        "current_node_id": None,
        "last_vlm_hypothesis": None,
        "rejection_reason": None
    }
    candidates = [{"node_id": 5, "score": 0.8}]
    
    hypothesis1 = generate_fallback_hypothesis(belief, candidates)
    hypothesis2 = generate_fallback_hypothesis(belief, candidates)
    
    assert hypothesis1 == hypothesis2

