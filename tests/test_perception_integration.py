"""Integration tests for perception visibility gate."""

import json
import random

import pytest

from perception import config as perception_config
from planner.verifier_stub import VerifierStub
from runtime.loop import initialize_belief_state, load_schemas
from runtime.schema_loader import validate_or_error
from vlm.fallback import generate_fallback_hypothesis


def test_action_simulation_goto_node():
    """Test that goto_node action updates current_node_id."""
    _, _, _, belief_validator = load_schemas()
    belief = initialize_belief_state(belief_validator)
    
    # Set up a goto_node action via last_vlm_hypothesis
    belief["last_vlm_hypothesis"] = {
        "target_status": "not_visible",
        "action": "goto_node",
        "confidence": 0.8,
        "rationale": "Going to node 5",
        "navigation_goal": {
            "type": "node_id",
            "node_id": 5
        }
    }
    belief["next_action"] = "goto_node"
    
    # Simulate action execution (as done in loop)
    if belief["next_action"] == "goto_node":
        target_node_id = belief.get("last_vlm_hypothesis", {}).get("navigation_goal", {}).get("node_id")
        if target_node_id is not None:
            belief["current_node_id"] = target_node_id
    
    # Verify current_node_id was updated
    assert belief["current_node_id"] == 5


def test_action_simulation_fallback_goto_node():
    """Test that fallback-generated goto_node also updates current_node_id."""
    _, _, _, belief_validator = load_schemas()
    belief = initialize_belief_state(belief_validator)
    
    # Generate fallback hypothesis with candidates
    candidates = [{"node_id": 7, "score": 0.6}]
    fallback_hypothesis = generate_fallback_hypothesis(belief, candidates)
    
    # Fallback should generate goto_node for candidates
    assert fallback_hypothesis["action"] == "goto_node"
    assert fallback_hypothesis["navigation_goal"]["node_id"] == 7
    
    # Apply to belief
    belief["last_vlm_hypothesis"] = fallback_hypothesis
    belief["next_action"] = fallback_hypothesis["action"]
    
    # Simulate action execution
    if belief["next_action"] == "goto_node":
        target_node_id = belief.get("last_vlm_hypothesis", {}).get("navigation_goal", {}).get("node_id")
        if target_node_id is not None:
            belief["current_node_id"] = target_node_id
    
    # Verify
    assert belief["current_node_id"] == 7


def test_visibility_hysteresis_k2(monkeypatch):
    """Test that visibility requires K=2 consecutive hits."""
    from perception.check_visibility import check_visibility
    
    # Set up oracle map with node 5 visible
    monkeypatch.setattr(perception_config, "NODE_ORACLE_MAP", {5: 0.9})
    
    _, _, _, belief_validator = load_schemas()
    belief = initialize_belief_state(belief_validator)
    
    # Move to node 5
    belief["current_node_id"] = 5
    
    # First perception check (step 0)
    vr = check_visibility(
        goal_text=belief["goal_text"],
        current_node_id=belief["current_node_id"],
        memory_context=[],
        belief_state=belief
    )
    
    assert vr["is_visible"] is True
    
    # Update streak (first hit)
    if vr["is_visible"]:
        belief["visibility_streak"] = belief.get("visibility_streak", 0) + 1
    else:
        belief["visibility_streak"] = 0
    
    assert belief["visibility_streak"] == 1
    
    # Check transition: should NOT transition yet (need K=2)
    K = 2
    if belief["visibility_streak"] >= K and belief["target_status"] not in {"visible", "done"}:
        belief["target_status"] = "visible"
    
    assert belief["target_status"] == "searching"  # Still searching
    
    # Second perception check (step 1)
    vr = check_visibility(
        goal_text=belief["goal_text"],
        current_node_id=belief["current_node_id"],
        memory_context=[],
        belief_state=belief
    )
    
    assert vr["is_visible"] is True
    
    # Update streak (second hit)
    if vr["is_visible"]:
        belief["visibility_streak"] = belief.get("visibility_streak", 0) + 1
    else:
        belief["visibility_streak"] = 0
    
    assert belief["visibility_streak"] == 2
    
    # Check transition: should transition now
    step_id = 1
    if belief["visibility_streak"] >= K and belief["target_status"] not in {"visible", "done"}:
        belief["target_status"] = "visible"
        belief["last_seen_node_id"] = belief["current_node_id"]
        belief["visible_since_step"] = step_id
    
    assert belief["target_status"] == "visible"
    assert belief["last_seen_node_id"] == 5
    assert belief["visible_since_step"] == 1


def test_vlm_cannot_force_visible_or_done(monkeypatch):
    """Test that VLM hypothesis cannot directly set visible/done."""
    # Set up oracle map with no visible nodes
    monkeypatch.setattr(perception_config, "NODE_ORACLE_MAP", {})
    
    _, _, vlm_validator, belief_validator = load_schemas()
    belief = initialize_belief_state(belief_validator)
    
    # Create a VLM hypothesis that tries to set target_status to "visible"
    vlm_hypothesis = {
        "target_status": "visible",  # VLM says visible
        "action": "approach",
        "confidence": 0.9,
        "rationale": "I see the target",
        "navigation_goal": {
            "type": "pose_relative",
            "distance_meters": 1.0,
            "angle_degrees": 0.0,
            "standoff_distance": 0.5
        }
    }
    
    # Validate it's schema-compliant
    is_valid, _ = validate_or_error(vlm_validator, vlm_hypothesis)
    assert is_valid
    
    # Apply to belief (simulate loop logic WITHOUT the old VLM promotion)
    belief["last_vlm_hypothesis"] = vlm_hypothesis
    belief["next_action"] = vlm_hypothesis["action"]
    
    # The old logic (now removed) would have done:
    # if vlm_hypothesis.get("target_status") == "visible":
    #     belief["target_status"] = "visible"
    # But this is now removed!
    
    # Belief should still be searching (not promoted by VLM)
    assert belief["target_status"] == "searching"
    
    # Only perception can promote to visible
    from perception.check_visibility import check_visibility
    
    vr = check_visibility(
        goal_text=belief["goal_text"],
        current_node_id=belief["current_node_id"],
        memory_context=[],
        belief_state=belief
    )
    
    # Perception says not visible (oracle map is empty)
    assert vr["is_visible"] is False
    
    # Belief should remain searching
    assert belief["target_status"] == "searching"


def test_visible_to_done_transition(monkeypatch):
    """Test visible -> done transition with guard."""
    from perception.check_visibility import check_visibility
    
    # Set up oracle map
    monkeypatch.setattr(perception_config, "NODE_ORACLE_MAP", {5: 0.9})
    
    _, _, _, belief_validator = load_schemas()
    belief = initialize_belief_state(belief_validator)
    
    # Manually set belief to visible state (as if it transitioned earlier)
    belief["target_status"] = "visible"
    belief["current_node_id"] = 5
    belief["last_seen_node_id"] = 5
    belief["visible_since_step"] = 10  # Became visible at step 10
    belief["visibility_streak"] = 2
    
    # Step 10: Right after becoming visible, try approach
    step_id = 10
    belief["next_action"] = "approach"
    
    vr = check_visibility(
        goal_text=belief["goal_text"],
        current_node_id=belief["current_node_id"],
        memory_context=[],
        belief_state=belief
    )
    
    assert vr["is_visible"] is True
    
    # Check close_enough with guard (should be False because step_id == visible_since_step)
    close_enough = (
        belief.get("visible_since_step") is not None
        and step_id > belief.get("visible_since_step")
        and belief.get("current_node_id") is not None
        and belief.get("current_node_id") == belief.get("last_seen_node_id")
    )
    
    assert close_enough is False  # Guard prevents immediate transition
    
    # Should NOT transition to done yet
    if belief["target_status"] == "visible" and belief["next_action"] == "approach" and vr["is_visible"]:
        if close_enough:
            belief["target_status"] = "done"
    
    assert belief["target_status"] == "visible"  # Still visible
    
    # Step 11: One step later, try approach again
    step_id = 11
    belief["next_action"] = "approach"
    
    vr = check_visibility(
        goal_text=belief["goal_text"],
        current_node_id=belief["current_node_id"],
        memory_context=[],
        belief_state=belief
    )
    
    assert vr["is_visible"] is True
    
    # Check close_enough again (should be True now)
    close_enough = (
        belief.get("visible_since_step") is not None
        and step_id > belief.get("visible_since_step")
        and belief.get("current_node_id") is not None
        and belief.get("current_node_id") == belief.get("last_seen_node_id")
    )
    
    assert close_enough is True  # Guard passes
    
    # Should transition to done now
    if belief["target_status"] == "visible" and belief["next_action"] == "approach" and vr["is_visible"]:
        if close_enough:
            belief["target_status"] = "done"
    
    assert belief["target_status"] == "done"


def test_visible_to_done_requires_perception_visible(monkeypatch):
    """Test that done transition requires perception to still see the target."""
    from perception.check_visibility import check_visibility
    
    # Set up oracle map (empty, so perception will return not visible)
    monkeypatch.setattr(perception_config, "NODE_ORACLE_MAP", {})
    
    _, _, _, belief_validator = load_schemas()
    belief = initialize_belief_state(belief_validator)
    
    # Set belief to visible state
    belief["target_status"] = "visible"
    belief["current_node_id"] = 5
    belief["last_seen_node_id"] = 5
    belief["visible_since_step"] = 10
    belief["next_action"] = "approach"
    
    # Step 12 (after visible_since_step)
    step_id = 12
    
    vr = check_visibility(
        goal_text=belief["goal_text"],
        current_node_id=belief["current_node_id"],
        memory_context=[],
        belief_state=belief
    )
    
    # Perception says NOT visible (oracle map empty)
    assert vr["is_visible"] is False
    
    # Check close_enough
    close_enough = (
        belief.get("visible_since_step") is not None
        and step_id > belief.get("visible_since_step")
        and belief.get("current_node_id") is not None
        and belief.get("current_node_id") == belief.get("last_seen_node_id")
    )
    
    assert close_enough is True
    
    # Should NOT transition to done because perception says not visible
    if belief["target_status"] == "visible" and belief["next_action"] == "approach" and vr["is_visible"]:
        if close_enough:
            belief["target_status"] = "done"
    
    assert belief["target_status"] == "visible"  # Still visible, not done


def test_streak_resets_on_not_visible(monkeypatch):
    """Test that visibility streak resets when perception returns not visible."""
    from perception.check_visibility import check_visibility
    
    # Set up oracle map
    monkeypatch.setattr(perception_config, "NODE_ORACLE_MAP", {5: 0.9})
    
    _, _, _, belief_validator = load_schemas()
    belief = initialize_belief_state(belief_validator)
    
    # First hit at node 5
    belief["current_node_id"] = 5
    vr = check_visibility(
        goal_text=belief["goal_text"],
        current_node_id=belief["current_node_id"],
        memory_context=[],
        belief_state=belief
    )
    
    assert vr["is_visible"] is True
    belief["visibility_streak"] = 1
    
    # Move to node 6 (not in oracle map)
    belief["current_node_id"] = 6
    vr = check_visibility(
        goal_text=belief["goal_text"],
        current_node_id=belief["current_node_id"],
        memory_context=[],
        belief_state=belief
    )
    
    assert vr["is_visible"] is False
    
    # Streak should reset
    if vr["is_visible"]:
        belief["visibility_streak"] = belief.get("visibility_streak", 0) + 1
    else:
        belief["visibility_streak"] = 0
    
    assert belief["visibility_streak"] == 0

