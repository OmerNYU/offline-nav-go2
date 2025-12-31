"""Tests for Ollama VLM client."""

import json
from unittest.mock import Mock, patch

from vlm.ollama_client import OllamaVLMClient


def test_extract_json_pure():
    """Test JSON extraction from pure JSON string."""
    client = OllamaVLMClient()
    
    json_str = '{"target_status": "visible", "action": "stop", "confidence": 0.9, "rationale": "Done"}'
    result = client._extract_json(json_str)
    
    assert result is not None
    assert result["target_status"] == "visible"
    assert result["action"] == "stop"


def test_extract_json_with_prose():
    """Test JSON extraction from text with prose."""
    client = OllamaVLMClient()
    
    text = 'Here is the result: {"target_status": "visible", "action": "stop", "confidence": 0.9, "rationale": "Done"}'
    result = client._extract_json(text)
    
    assert result is not None
    assert result["target_status"] == "visible"


def test_extract_json_code_fence():
    """Test JSON extraction from code fence."""
    client = OllamaVLMClient()
    
    text = '''```json
{"target_status": "visible", "action": "stop", "confidence": 0.9, "rationale": "Done"}
```'''
    result = client._extract_json(text)
    
    assert result is not None
    assert result["target_status"] == "visible"


def test_extract_json_multiple_objects():
    """Test JSON extraction extracts first complete object."""
    client = OllamaVLMClient()
    
    text = '{"target_status": "visible", "action": "stop", "confidence": 0.9, "rationale": "First"} {"other": "data"}'
    result = client._extract_json(text)
    
    assert result is not None
    assert result["rationale"] == "First"


def test_extract_json_balanced_braces_with_extra_text():
    """Test balanced brace extraction with extra text."""
    client = OllamaVLMClient()
    
    text = 'Some text { "nested": {"target_status": "visible"} } more text'
    result = client._extract_json(text)
    
    assert result is not None
    # Should extract the outer object
    assert "nested" in result


def test_extract_json_braces_in_string():
    """Test that braces inside string fields don't break extraction."""
    client = OllamaVLMClient()
    
    text = '{"target_status": "visible", "action": "stop", "confidence": 0.9, "rationale": "Text with {braces} inside"}'
    result = client._extract_json(text)
    
    assert result is not None
    assert result["rationale"] == "Text with {braces} inside"


def test_extract_json_invalid():
    """Test JSON extraction returns None for invalid JSON."""
    client = OllamaVLMClient()
    
    text = '{ this is not json'
    result = client._extract_json(text)
    
    assert result is None


def test_extract_json_empty():
    """Test JSON extraction returns None for empty string."""
    client = OllamaVLMClient()
    
    result = client._extract_json("")
    
    assert result is None


def test_validation_gate_invalid_output():
    """Test that invalid model output is rejected."""
    client = OllamaVLMClient()
    
    # Mock the API call to return invalid JSON (missing required fields)
    with patch.object(client, '_call_ollama_api') as mock_api:
        mock_api.return_value = ('{"target_status": "visible", "action": "stop"}', None)
        
        context = {
            "goal_text": "test",
            "active_constraints": [],
            "belief_target_status": "searching",
            "candidate_nodes": [],
            "memory_context": []
        }
        
        hypothesis, meta = client.propose_hypothesis(context)
        
        assert hypothesis is None
        assert meta["vlm_schema_ok"] is False


def test_validation_gate_valid_output():
    """Test that valid model output is accepted."""
    client = OllamaVLMClient()
    
    valid_json = json.dumps({
        "target_status": "not_visible",
        "action": "explore",
        "confidence": 0.6,
        "rationale": "Searching for target"
    })
    
    with patch.object(client, '_call_ollama_api') as mock_api:
        mock_api.return_value = (valid_json, None)
        
        context = {
            "goal_text": "test",
            "active_constraints": [],
            "belief_target_status": "searching",
            "candidate_nodes": [],
            "memory_context": []
        }
        
        hypothesis, meta = client.propose_hypothesis(context)
        
        assert hypothesis is not None
        assert meta["vlm_schema_ok"] is True
        assert hypothesis["action"] == "explore"


def test_retry_logic():
    """Test retry logic on invalid JSON."""
    client = OllamaVLMClient(max_retries=1)
    
    valid_json = json.dumps({
        "target_status": "not_visible",
        "action": "explore",
        "confidence": 0.6,
        "rationale": "Searching"
    })
    
    with patch.object(client, '_call_ollama_api') as mock_api:
        # First call returns invalid JSON, second call returns valid
        mock_api.side_effect = [
            ('invalid json {', None),
            (valid_json, None)
        ]
        
        context = {
            "goal_text": "test",
            "active_constraints": [],
            "belief_target_status": "searching",
            "candidate_nodes": [],
            "memory_context": []
        }
        
        hypothesis, meta = client.propose_hypothesis(context)
        
        assert hypothesis is not None
        assert meta["vlm_retry_count"] == 1
        assert meta["vlm_schema_ok"] is True


def test_ollama_unavailable():
    """Test handling when Ollama is unavailable."""
    client = OllamaVLMClient()
    
    with patch.object(client, '_call_ollama_api') as mock_api:
        mock_api.return_value = (None, "connection_failed:test error")
        
        context = {
            "goal_text": "test",
            "active_constraints": [],
            "belief_target_status": "searching",
            "candidate_nodes": [],
            "memory_context": []
        }
        
        hypothesis, meta = client.propose_hypothesis(context)
        
        assert hypothesis is None
        assert meta["vlm_error"] is not None
        assert "connection_failed" in meta["vlm_error"]


def test_ollama_error_payload():
    """Test handling of Ollama error payloads."""
    client = OllamaVLMClient()
    
    with patch.object(client, '_call_ollama_api') as mock_api:
        mock_api.return_value = (None, "ollama_error:model not found")
        
        context = {
            "goal_text": "test",
            "active_constraints": [],
            "belief_target_status": "searching",
            "candidate_nodes": [],
            "memory_context": []
        }
        
        hypothesis, meta = client.propose_hypothesis(context)
        
        assert hypothesis is None
        assert "ollama_error" in meta["vlm_error"]

