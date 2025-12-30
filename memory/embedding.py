"""Deterministic text embedding for semantic memory."""

import hashlib
import math
import random


def normalize(vec: list[float]) -> list[float]:
    """Normalize a vector to unit length.
    
    Args:
        vec: Input vector.
        
    Returns:
        Normalized vector with unit length.
    """
    magnitude = math.sqrt(sum(x * x for x in vec))
    if magnitude == 0.0:
        return vec
    return [x / magnitude for x in vec]


class DeterministicEmbedder:
    """Deterministic text embedder using hash-seeded random vectors.
    
    Produces consistent embeddings for the same input text.
    """
    
    def __init__(self, dim: int = 64) -> None:
        """Initialize embedder with specified dimensionality.
        
        Args:
            dim: Embedding dimension (default 64).
        """
        self.dim = dim
    
    def embed_text(self, text: str) -> list[float]:
        """Generate deterministic embedding for text.
        
        Uses MD5 hash of normalized text to seed RNG, then generates
        a random vector and normalizes it to unit length.
        
        Args:
            text: Input text to embed.
            
        Returns:
            Normalized embedding vector of length self.dim.
        """
        # Normalize text
        normalized = text.lower().strip()
        
        # Use stable hash to seed RNG
        text_hash = hashlib.md5(normalized.encode('utf-8')).hexdigest()
        seed = int(text_hash[:8], 16)  # Use first 8 hex chars as seed
        
        # Generate deterministic vector
        rng = random.Random(seed)
        vec = [rng.gauss(0, 1) for _ in range(self.dim)]
        
        # Return normalized vector
        return normalize(vec)

