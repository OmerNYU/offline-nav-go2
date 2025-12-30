"""Utility functions for semantic memory processing."""

import re


def tokenize(text: str) -> list[str]:
    """Tokenize text with robust normalization.
    
    Normalizes text by:
    - Converting to lowercase
    - Stripping whitespace
    - Removing punctuation
    - Splitting on whitespace
    - Filtering out empty tokens
    
    Args:
        text: Input text to tokenize.
        
    Returns:
        List of non-empty tokens.
    """
    # Lowercase and strip
    normalized = text.lower().strip()
    
    # Remove punctuation
    no_punct = re.sub(r'[^\w\s]', '', normalized)
    
    # Split on whitespace and filter empty strings
    tokens = [token for token in no_punct.split() if token]
    
    return tokens

