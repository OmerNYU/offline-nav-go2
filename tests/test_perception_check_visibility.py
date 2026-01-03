"""Tests for perception visibility checking."""

import pytest

from perception import config as perception_config
from perception.check_visibility import check_visibility


def test_check_visibility_node_oracle_map(monkeypatch):
    """Test that oracle map lookup works correctly."""
    # Set up oracle map
    test_oracle_map = {5: 0.9, 7: 0.75}
    monkeypatch.setattr(perception_config, "NODE_ORACLE_MAP", test_oracle_map)
    
    # Test case 1: node in oracle map
    result = check_visibility(
        goal_text="red backpack",
        current_node_id=5,
        memory_context=[],
        belief_state={}
    )
    
    assert result["is_visible"] is True
    assert result["confidence"] == 0.9
    assert result["backend"] == "node_oracle"
    assert "latency_ms" in result
    assert isinstance(result["latency_ms"], int)
    assert result["evidence"]["reason"] == "oracle_hit"
    assert result["evidence"]["node_id"] == 5
    assert result["evidence"]["extra"]["oracle_confidence"] == 0.9
    
    # Test case 2: node not in oracle map
    result = check_visibility(
        goal_text="red backpack",
        current_node_id=6,
        memory_context=[],
        belief_state={}
    )
    
    assert result["is_visible"] is False
    assert result["confidence"] == 0.0
    assert result["backend"] == "node_oracle"
    assert result["evidence"]["reason"] == "oracle_miss"
    assert result["evidence"]["node_id"] == 6
    
    # Test case 3: current_node_id is None
    result = check_visibility(
        goal_text="red backpack",
        current_node_id=None,
        memory_context=[],
        belief_state={}
    )
    
    assert result["is_visible"] is False
    assert result["confidence"] == 0.0
    assert result["evidence"]["reason"] == "oracle_miss"
    assert result["evidence"]["node_id"] is None


def test_check_visibility_error_handling():
    """Test that exceptions are handled safely."""
    # Create a malformed config that will cause an exception
    bad_config = {"oracle_map": "not_a_dict"}  # This will cause TypeError
    
    # Should not raise exception, but return safe default
    result = check_visibility(
        goal_text="red backpack",
        current_node_id=5,
        memory_context=[],
        belief_state={},
        config=bad_config
    )
    
    assert result["is_visible"] is False
    assert result["confidence"] == 0.0
    assert result["evidence"]["reason"] == "exception"
    assert "error" in result["evidence"]["extra"]
    assert "exception_type" in result["evidence"]["extra"]


def test_check_visibility_backend_override():
    """Test that backend can be overridden."""
    result = check_visibility(
        goal_text="red backpack",
        current_node_id=5,
        memory_context=[],
        belief_state={},
        backend="unknown_backend"
    )
    
    assert result["backend"] == "unknown_backend"
    assert result["is_visible"] is False
    assert result["evidence"]["reason"] == "unknown_backend"


def test_check_visibility_config_override(monkeypatch):
    """Test that config override works (for testing)."""
    # Set global map to something different
    monkeypatch.setattr(perception_config, "NODE_ORACLE_MAP", {1: 0.5})
    
    # Override with custom config
    custom_oracle_map = {10: 0.95}
    result = check_visibility(
        goal_text="red backpack",
        current_node_id=10,
        memory_context=[],
        belief_state={},
        config={"oracle_map": custom_oracle_map}
    )
    
    assert result["is_visible"] is True
    assert result["confidence"] == 0.95
    assert result["evidence"]["reason"] == "oracle_hit"


def test_get_node_oracle_map_returns_copy(monkeypatch):
    """Test that get_node_oracle_map returns a copy to prevent mutation."""
    from perception.config import get_node_oracle_map
    
    # Set up oracle map
    test_oracle_map = {5: 0.9}
    monkeypatch.setattr(perception_config, "NODE_ORACLE_MAP", test_oracle_map)
    
    # Get copy
    oracle_copy = get_node_oracle_map()
    
    # Mutate copy
    oracle_copy[99] = 0.5
    
    # Original should be unchanged
    assert 99 not in perception_config.NODE_ORACLE_MAP
    assert perception_config.NODE_ORACLE_MAP == {5: 0.9}


def test_node_oracle_relpose_backend_hit():
    """Test relpose backend returns distance/bearing when mapped."""
    # TWEAK 2: Use config injection (no monkeypatch)
    test_relpose_map = {
        "find the red backpack": {
            12: {"distance_m": 1.8, "bearing_rad": 0.2, "confidence": 0.9}
        }
    }
    
    result = check_visibility(
        goal_text="Find the RED BACKPACK",  # Test normalization
        current_node_id=12,
        memory_context=[],
        belief_state={},
        backend="node_oracle_relpose",
        config={"relpose_map": test_relpose_map}
    )
    
    assert result["is_visible"] is True
    assert result["confidence"] == 0.9
    assert result["distance_m"] == 1.8
    assert result["bearing_rad"] == 0.2
    assert result["backend"] == "node_oracle_relpose"
    assert result["target_goal_key"] == "find the red backpack"
    assert result["evidence"]["reason"] == "relpose_hit"


def test_node_oracle_relpose_backend_miss():
    """Test relpose backend returns not visible when not mapped."""
    test_relpose_map = {
        "other goal": {
            99: {"distance_m": 1.0, "bearing_rad": 0.0, "confidence": 1.0}
        }
    }
    
    result = check_visibility(
        goal_text="find the red backpack",
        current_node_id=12,
        memory_context=[],
        belief_state={},
        backend="node_oracle_relpose",
        config={"relpose_map": test_relpose_map}
    )
    
    assert result["is_visible"] is False
    assert result["confidence"] == 0.0
    # TWEAK 3: Verify explicit None assignment
    assert result["distance_m"] is None
    assert result["bearing_rad"] is None
    assert result["evidence"]["reason"] == "relpose_miss"


def test_node_oracle_relpose_confidence_clamping():
    """Test confidence clamping to [0.0, 1.0] (FIX D)."""
    test_relpose_map = {
        "test goal": {
            10: {"distance_m": 1.0, "bearing_rad": 0.0, "confidence": 1.5},  # > 1.0
            11: {"distance_m": 1.0, "bearing_rad": 0.0, "confidence": -0.5}, # < 0.0
        }
    }
    
    # Test upper clamp
    result = check_visibility(
        goal_text="test goal", current_node_id=10,
        memory_context=[], belief_state={},
        backend="node_oracle_relpose",
        config={"relpose_map": test_relpose_map}
    )
    assert result["confidence"] == 1.0  # Clamped from 1.5
    
    # Test lower clamp
    result = check_visibility(
        goal_text="test goal", current_node_id=11,
        memory_context=[], belief_state={},
        backend="node_oracle_relpose",
        config={"relpose_map": test_relpose_map}
    )
    assert result["confidence"] == 0.0  # Clamped from -0.5


def test_node_oracle_relpose_normalize_empty_goal():
    """Test normalize_goal_text handles None/empty strings (TWEAK 4)."""
    from perception.config import normalize_goal_text
    
    assert normalize_goal_text("") == ""
    assert normalize_goal_text(None) == ""
    assert normalize_goal_text("  ") == ""
    assert normalize_goal_text("Find   the    RED  backpack  ") == "find the red backpack"


def test_backward_compatibility_distance_none():
    """Test node_oracle backend returns None for distance/bearing."""
    result = check_visibility(
        goal_text="test",
        current_node_id=5,
        memory_context=[],
        belief_state={},
        backend="node_oracle",
        config={"oracle_map": {5: 0.9}}
    )
    
    assert result["is_visible"] is True  # Standard visibility works
    assert result["distance_m"] is None  # No distance for classic backend
    assert result["bearing_rad"] is None
    assert result["target_goal_key"] is None
