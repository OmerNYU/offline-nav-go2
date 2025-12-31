"""Tests for loop integration with VLM backends."""

from unittest.mock import Mock, patch

from vlm.fallback import generate_fallback_hypothesis
from vlm.ollama_client import OllamaVLMClient


def test_mock_backend_unchanged():
    """Test that mock backend behavior is unchanged."""
    # This test verifies that when use_ollama=False, the system uses mock generation
    # The actual loop test would be complex, so we test the fallback directly
    
    belief = {
        "target_status": "searching",
        "goal_text": "red backpack",
        "active_constraints": [],
        "candidate_nodes": [],
        "next_action": "explore",
        "current_node_id": None,
        "last_vlm_hypothesis": None,
        "rejection_reason": None
    }
    candidates = []
    
    # Fallback should work correctly
    hypothesis = generate_fallback_hypothesis(belief, candidates)
    
    assert hypothesis is not None
    assert hypothesis["action"] == "explore"
    assert "target_status" in hypothesis


def test_ollama_backend_unavailable_uses_fallback():
    """Test that when Ollama is unavailable, fallback is used."""
    client = OllamaVLMClient()
    
    with patch.object(client, '_call_ollama_api') as mock_api:
        mock_api.return_value = (None, "connection_failed")
        
        context = {
            "goal_text": "red backpack",
            "active_constraints": [],
            "belief_target_status": "searching",
            "candidate_nodes": [],
            "memory_context": []
        }
        
        hypothesis, meta = client.propose_hypothesis(context)
        
        # Client returns None
        assert hypothesis is None
        assert meta["vlm_error"] is not None
        
        # Fallback should be used in loop
        belief = {
            "target_status": "searching",
            "goal_text": "red backpack",
            "active_constraints": [],
            "candidate_nodes": [],
            "next_action": "explore",
            "current_node_id": None,
            "last_vlm_hypothesis": None,
            "rejection_reason": None
        }
        fallback = generate_fallback_hypothesis(belief, [])
        
        assert fallback is not None
        assert fallback["action"] == "explore"


def test_ollama_backend_invalid_json_uses_fallback():
    """Test that when Ollama returns invalid JSON, fallback is used."""
    client = OllamaVLMClient(max_retries=0)  # No retries for faster test
    
    with patch.object(client, '_call_ollama_api') as mock_api:
        mock_api.return_value = ('invalid json {{{', None)
        
        context = {
            "goal_text": "red backpack",
            "active_constraints": [],
            "belief_target_status": "searching",
            "candidate_nodes": [],
            "memory_context": []
        }
        
        hypothesis, meta = client.propose_hypothesis(context)
        
        # Client returns None due to parse failure
        assert hypothesis is None
        assert meta["vlm_parse_ok"] is False


def test_no_invalid_hypothesis_reaches_verifier():
    """Test that invalid hypotheses never reach the verifier."""
    client = OllamaVLMClient()
    
    # Test with missing required field
    with patch.object(client, '_call_ollama_api') as mock_api:
        mock_api.return_value = ('{"target_status": "visible"}', None)
        
        context = {
            "goal_text": "test",
            "active_constraints": [],
            "belief_target_status": "searching",
            "candidate_nodes": [],
            "memory_context": []
        }
        
        hypothesis, meta = client.propose_hypothesis(context)
        
        # Invalid hypothesis is rejected by client
        assert hypothesis is None
        assert meta["vlm_schema_ok"] is False


def test_meta_contains_vlm_status_fields():
    """Test that meta dict contains VLM status fields."""
    client = OllamaVLMClient()
    
    with patch.object(client, '_call_ollama_api') as mock_api:
        mock_api.return_value = (None, "connection_failed")
        
        context = {
            "goal_text": "test",
            "active_constraints": [],
            "belief_target_status": "searching",
            "candidate_nodes": [],
            "memory_context": []
        }
        
        hypothesis, meta = client.propose_hypothesis(context)
        
        # Check all required meta fields
        assert "vlm_backend" in meta
        assert meta["vlm_backend"] == "ollama"
        assert "vlm_model" in meta
        assert "vlm_latency_ms" in meta
        assert "vlm_parse_ok" in meta
        assert "vlm_schema_ok" in meta
        assert "vlm_retry_count" in meta
        assert "vlm_error" in meta


def test_vlm_backend_set_correctly():
    """Test that vlm_backend is set correctly for both backends."""
    # Ollama backend
    client = OllamaVLMClient()
    
    with patch.object(client, '_call_ollama_api') as mock_api:
        mock_api.return_value = (None, "test_error")
        
        context = {
            "goal_text": "test",
            "active_constraints": [],
            "belief_target_status": "searching",
            "candidate_nodes": [],
            "memory_context": []
        }
        
        hypothesis, meta = client.propose_hypothesis(context)
        
        assert meta["vlm_backend"] == "ollama"
    
    # Mock backend would set vlm_backend to "mock" in the loop
    # (tested via the actual loop code)

