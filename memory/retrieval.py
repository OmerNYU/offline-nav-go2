"""Semantic retrieval for memory-guided navigation."""

import math
from typing import Optional

from memory.embedding import DeterministicEmbedder
from memory.store import SemanticMemoryStore
from memory.utils import tokenize


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector.
        vec2: Second vector.
        
    Returns:
        Cosine similarity in range [-1, 1].
    """
    if len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    
    if magnitude1 == 0.0 or magnitude2 == 0.0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


def retrieve_candidates(
    goal_text: str,
    store: SemanticMemoryStore,
    embedder: DeterministicEmbedder,
    k: int = 5
) -> list[dict]:
    """Retrieve top-k memory nodes matching goal_text.
    
    Uses hybrid scoring: 0.8 * embedding_similarity + 0.2 * keyword_overlap.
    Deterministic ordering with explicit tie-breaking by node_id.
    
    Args:
        goal_text: Natural language goal description.
        store: Semantic memory store to search.
        embedder: Text embedder for computing similarity.
        k: Maximum number of candidates to return (default 5).
        
    Returns:
        List of candidates sorted by score (descending), each with:
        - node_id: int
        - score: float in [0.0, 1.0]
    """
    # Tokenize goal text
    goal_tokens = tokenize(goal_text)
    
    # If no valid tokens, return empty list immediately
    if not goal_tokens:
        return []
    
    # Embed goal text
    goal_embedding = embedder.embed_text(goal_text)
    goal_token_set = set(goal_tokens)
    
    # Get all nodes and sort by node_id for determinism
    nodes = store.all_nodes()
    nodes_sorted = sorted(nodes, key=lambda n: n.node_id)
    
    # Score each node
    candidates = []
    for node in nodes_sorted:
        # Compute embedding similarity
        if node.embedding is not None:
            node_vec = node.embedding
        else:
            # Use embedding of summary + tags
            node_text = node.summary + " " + " ".join(node.tags)
            node_vec = embedder.embed_text(node_text)
        
        cos_sim = cosine_similarity(goal_embedding, node_vec)
        # Map cosine [-1, 1] to [0, 1]
        embedding_score = (cos_sim + 1.0) / 2.0
        
        # Compute keyword overlap
        node_text = node.summary + " " + " ".join(node.tags)
        node_tokens = set(tokenize(node_text))
        overlap = len(goal_token_set & node_tokens)
        keyword_score = overlap / len(goal_tokens)  # Safe: goal_tokens is non-empty
        
        # Blend scores: 0.8 embedding + 0.2 keywords
        final_score = 0.8 * embedding_score + 0.2 * keyword_score
        
        # Clamp to [0, 1] for numeric safety
        final_score = max(0.0, min(1.0, final_score))
        
        # Ensure score is Python float (not numpy)
        final_score = float(final_score)
        
        candidates.append({
            "node_id": node.node_id,
            "score": final_score
        })
    
    # Sort by score (descending) with tie-break by node_id (ascending)
    candidates_sorted = sorted(candidates, key=lambda c: (-c["score"], c["node_id"]))
    
    # Return top-k
    return candidates_sorted[:k]

