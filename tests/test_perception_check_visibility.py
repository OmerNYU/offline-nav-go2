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

