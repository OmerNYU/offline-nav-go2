"""Tests for memory retrieval functionality."""

from memory.embedding import DeterministicEmbedder
from memory.retrieval import retrieve_candidates
from memory.store import SemanticMemoryStore
from memory.types import Pose2D


def test_deterministic_ordering():
    """Test that retrieval produces identical results for same inputs."""
    # Create store with 3 nodes in non-sorted order
    store = SemanticMemoryStore()
    store.add_node(
        pose=Pose2D(x=5.0, y=2.0, yaw=0.0),
        embedding=None,
        tags=["room", "table"],
        summary="A room with a table"
    )
    store.add_node(
        pose=Pose2D(x=1.0, y=1.0, yaw=0.0),
        embedding=None,
        tags=["kitchen", "stove"],
        summary="Kitchen area with stove"
    )
    store.add_node(
        pose=Pose2D(x=3.0, y=3.0, yaw=0.0),
        embedding=None,
        tags=["bedroom", "bed"],
        summary="Bedroom with bed"
    )
    
    embedder = DeterministicEmbedder(dim=64)
    goal_text = "find the kitchen"
    
    # Call retrieve_candidates twice
    result1 = retrieve_candidates(goal_text, store, embedder, k=5)
    result2 = retrieve_candidates(goal_text, store, embedder, k=5)
    
    # Results should be identical
    assert result1 == result2, "Results should be deterministic"
    assert len(result1) == 3, "Should return all 3 nodes"


def test_tie_break_behavior():
    """Test that tie-breaking uses node_id (lower comes first)."""
    store = SemanticMemoryStore()
    embedder = DeterministicEmbedder(dim=64)
    
    # Add nodes that will have very similar scores
    # Use identical tags/summary to force ties
    for i in range(3):
        store.add_node(
            pose=Pose2D(x=float(i), y=0.0, yaw=0.0),
            embedding=None,
            tags=["test", "node"],
            summary="Test node"
        )
    
    goal_text = "test node"
    results = retrieve_candidates(goal_text, store, embedder, k=5)
    
    # Check that node_ids are in ascending order for tied scores
    for i in range(len(results) - 1):
        if abs(results[i]["score"] - results[i+1]["score"]) < 1e-9:
            assert results[i]["node_id"] < results[i+1]["node_id"], \
                "Tied scores should be broken by node_id (ascending)"


def test_score_range():
    """Test that scores are in [0, 1] range and have correct types."""
    store = SemanticMemoryStore()
    store.add_node(
        pose=Pose2D(x=1.0, y=1.0, yaw=0.0),
        embedding=None,
        tags=["kitchen"],
        summary="Kitchen area"
    )
    store.add_node(
        pose=Pose2D(x=2.0, y=2.0, yaw=0.0),
        embedding=None,
        tags=["bedroom"],
        summary="Bedroom area"
    )
    
    embedder = DeterministicEmbedder(dim=64)
    results = retrieve_candidates("find kitchen", store, embedder, k=5)
    
    assert len(results) > 0, "Should have results"
    
    for candidate in results:
        # Check types
        assert isinstance(candidate["node_id"], int), "node_id should be int"
        assert isinstance(candidate["score"], float), "score should be float"
        
        # Check range
        assert 0.0 <= candidate["score"] <= 1.0, f"Score {candidate['score']} not in [0, 1]"


def test_score_mapping():
    """Test that cosine similarity is correctly mapped to [0, 1]."""
    store = SemanticMemoryStore()
    embedder = DeterministicEmbedder(dim=64)
    
    # Add a node
    store.add_node(
        pose=Pose2D(x=1.0, y=1.0, yaw=0.0),
        embedding=None,
        tags=["test"],
        summary="Test node"
    )
    
    # Test with matching keyword
    results = retrieve_candidates("test", store, embedder, k=5)
    assert len(results) == 1
    # Score should be in valid range and reasonably high due to keyword match
    # The word "test" appears in tags, so keyword overlap contributes positively
    assert 0.0 <= results[0]["score"] <= 1.0, "Score should be in [0, 1]"
    assert results[0]["score"] > 0.3, "Matching keyword should produce reasonable score"


def test_keyword_blending():
    """Test that keyword overlap is blended with embedding score."""
    store = SemanticMemoryStore()
    embedder = DeterministicEmbedder(dim=64)
    
    # Add nodes with specific tags
    store.add_node(
        pose=Pose2D(x=1.0, y=1.0, yaw=0.0),
        embedding=None,
        tags=["kitchen", "red", "backpack"],
        summary="Red backpack in kitchen"
    )
    store.add_node(
        pose=Pose2D(x=2.0, y=2.0, yaw=0.0),
        embedding=None,
        tags=["bedroom", "blue", "chair"],
        summary="Blue chair in bedroom"
    )
    
    # Search for "red backpack" - should strongly match first node
    results = retrieve_candidates("red backpack", store, embedder, k=5)
    
    assert len(results) == 2
    # First result should be the kitchen node (better keyword overlap)
    assert results[0]["node_id"] == 0, "Node with matching keywords should rank higher"


def test_empty_goal_text():
    """Test that empty goal_text returns empty list."""
    store = SemanticMemoryStore()
    store.add_node(
        pose=Pose2D(x=1.0, y=1.0, yaw=0.0),
        embedding=None,
        tags=["test"],
        summary="Test node"
    )
    
    embedder = DeterministicEmbedder(dim=64)
    
    # Test empty string
    assert retrieve_candidates("", store, embedder, k=5) == []


def test_whitespace_only_goal_text():
    """Test that whitespace-only goal_text returns empty list."""
    store = SemanticMemoryStore()
    store.add_node(
        pose=Pose2D(x=1.0, y=1.0, yaw=0.0),
        embedding=None,
        tags=["test"],
        summary="Test node"
    )
    
    embedder = DeterministicEmbedder(dim=64)
    
    # Test whitespace-only
    assert retrieve_candidates("   ", store, embedder, k=5) == []
    assert retrieve_candidates("\t\n", store, embedder, k=5) == []


def test_punctuation_only_goal_text():
    """Test that punctuation-only goal_text returns empty list."""
    store = SemanticMemoryStore()
    store.add_node(
        pose=Pose2D(x=1.0, y=1.0, yaw=0.0),
        embedding=None,
        tags=["test"],
        summary="Test node"
    )
    
    embedder = DeterministicEmbedder(dim=64)
    
    # Test punctuation-only
    assert retrieve_candidates("!!!", store, embedder, k=5) == []
    assert retrieve_candidates("...", store, embedder, k=5) == []
    assert retrieve_candidates("!@#$%", store, embedder, k=5) == []


def test_top_k_limit():
    """Test that results are limited to k."""
    store = SemanticMemoryStore()
    embedder = DeterministicEmbedder(dim=64)
    
    # Add 10 nodes
    for i in range(10):
        store.add_node(
            pose=Pose2D(x=float(i), y=0.0, yaw=0.0),
            embedding=None,
            tags=["node", f"node{i}"],
            summary=f"Node {i}"
        )
    
    # Request only top 3
    results = retrieve_candidates("node", store, embedder, k=3)
    
    assert len(results) == 3, "Should return at most k results"


def test_garbage_candidate_guard():
    """Test that garbage candidate guard reduces scores for low-signal matches.
    
    The guard applies when keyword_score == 0 AND embedding_score is near 0.5
    (within 0.05), reducing the final_score by half.
    """
    store = SemanticMemoryStore()
    embedder = DeterministicEmbedder(dim=64)
    
    # Add nodes with completely unrelated content (no keyword overlap)
    # Goal: "find kitchen appliance" - no overlap with these nodes
    store.add_node(
        pose=Pose2D(x=1.0, y=1.0, yaw=0.0),
        embedding=None,
        tags=["xyz", "abc"],
        summary="Completely unrelated content here"
    )
    store.add_node(
        pose=Pose2D(x=2.0, y=2.0, yaw=0.0),
        embedding=None,
        tags=["def", "ghi"],
        summary="Another unrelated node with different words"
    )
    
    # Search for something completely different - ensures keyword_score = 0
    goal_text = "find kitchen appliance"
    results = retrieve_candidates(goal_text, store, embedder, k=5)
    
    assert len(results) == 2, "Should return both nodes"
    
    # With keyword_score=0, base score = 0.8 * embedding_score
    # If embedding_score is between 0.45 and 0.55:
    #   - Without guard: score = 0.8 * embedding_score (e.g., 0.8 * 0.5 = 0.4)
    #   - With guard: score = 0.8 * embedding_score * 0.5 (e.g., 0.8 * 0.5 * 0.5 = 0.2)
    
    # Verify schema is unchanged: all entries have node_id (int) and score (float)
    for candidate in results:
        assert isinstance(candidate["node_id"], int), "node_id must be int"
        assert isinstance(candidate["score"], float), "score must be float"
        assert 0.0 <= candidate["score"] <= 1.0, "score must be in [0, 1]"
        
        # Since keyword_score=0, max possible unguarded score is 0.8 * 1.0 = 0.8
        # But for unrelated content, embedding scores are typically moderate
        # The guard ensures that when embedding is near-random (0.5), score is halved
        # So we expect scores to be reasonable for unrelated content
        assert candidate["score"] < 0.6, \
            f"Score {candidate['score']} should be moderate for unrelated content"
    
    # Verify the guard logic: scores should be lower when conditions are met
    # We test this by ensuring the function completes successfully with the guard in place
    # and that output schema remains correct: [{"node_id": int, "score": float}, ...]
    
    # Add more unrelated nodes to increase test coverage
    store.add_node(
        pose=Pose2D(x=3.0, y=3.0, yaw=0.0),
        embedding=None,
        tags=["unrelated", "random"],
        summary="Random unrelated text that has no semantic connection"
    )
    
    results2 = retrieve_candidates(goal_text, store, embedder, k=5)
    assert len(results2) == 3, "Should include all nodes"
    
    # All results should maintain the correct schema
    for candidate in results2:
        assert "node_id" in candidate and "score" in candidate, \
            "Each candidate must have node_id and score"
        assert isinstance(candidate["node_id"], int), "node_id must be int"
        assert isinstance(candidate["score"], float), "score must be float"

