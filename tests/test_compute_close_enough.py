"""Unit tests for compute_close_enough helper."""

from runtime.loop import compute_close_enough, CLOSE_ENOUGH_M


def test_compute_close_enough_relpose_within_threshold():
    """Test relpose backend with distance <= threshold."""
    belief = {"visible_since_step": 10}
    vr = {"distance_m": 0.5, "backend": "node_oracle_relpose"}
    
    result = compute_close_enough(belief, vr, step_id=11, close_enough_m=0.75, backend="node_oracle_relpose")
    
    assert result is True


def test_compute_close_enough_relpose_exceeds_threshold():
    """Test relpose backend with distance > threshold."""
    belief = {"visible_since_step": 10}
    vr = {"distance_m": 2.0, "backend": "node_oracle_relpose"}
    
    result = compute_close_enough(belief, vr, step_id=11, close_enough_m=0.75, backend="node_oracle_relpose")
    
    assert result is False


def test_compute_close_enough_relpose_missing_distance():
    """Test relpose backend with missing distance_m (FIX F)."""
    belief = {"visible_since_step": 10}
    vr = {"distance_m": None, "backend": "node_oracle_relpose"}
    
    result = compute_close_enough(belief, vr, step_id=11, close_enough_m=0.75, backend="node_oracle_relpose")
    
    assert result is False  # Explicit policy: missing distance → not close


def test_compute_close_enough_relpose_invalid_distance_type():
    """Test relpose backend with invalid distance type (TWEAK 5)."""
    belief = {"visible_since_step": 10}
    vr = {"distance_m": "not_a_number", "backend": "node_oracle_relpose"}
    
    result = compute_close_enough(belief, vr, step_id=11, close_enough_m=0.75, backend="node_oracle_relpose")
    
    assert result is False  # Type error caught → safe default


def test_compute_close_enough_guard_same_step():
    """Test visible_since_step guard prevents same-step done."""
    belief = {"visible_since_step": 10}
    vr = {"distance_m": 0.1, "backend": "node_oracle_relpose"}
    
    # Same step: should fail guard
    result = compute_close_enough(belief, vr, step_id=10, close_enough_m=0.75, backend="node_oracle_relpose")
    assert result is False
    
    # Next step: should pass
    result = compute_close_enough(belief, vr, step_id=11, close_enough_m=0.75, backend="node_oracle_relpose")
    assert result is True


def test_compute_close_enough_node_oracle_compat():
    """Test node_oracle backend uses node-based check."""
    belief = {
        "visible_since_step": 10,
        "current_node_id": 5,
        "last_seen_node_id": 5
    }
    vr = {"backend": "node_oracle"}
    
    result = compute_close_enough(belief, vr, step_id=11, close_enough_m=0.75, backend="node_oracle")
    
    assert result is True  # Same node → close enough


def test_compute_close_enough_node_oracle_different_node():
    """Test node_oracle backend requires same node."""
    belief = {
        "visible_since_step": 10,
        "current_node_id": 5,
        "last_seen_node_id": 6
    }
    vr = {"backend": "node_oracle"}
    
    result = compute_close_enough(belief, vr, step_id=11, close_enough_m=0.75, backend="node_oracle")
    
    assert result is False  # Different node → not close

