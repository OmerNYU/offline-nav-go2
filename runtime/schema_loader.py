"""Schema loading and validation for VLM hypothesis.

Provides Windows-safe schema loading with $ref resolution.
Extracted from runtime/loop.py to avoid circular imports.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from jsonschema import Draft7Validator, RefResolver


def load_hypothesis_validator() -> Draft7Validator:
    """Load VLM hypothesis validator with Windows-compatible $ref resolution.
    
    Returns:
        Draft7Validator configured for vlm_hypothesis.schema.json
    """
    schema_dir = Path("schema").resolve()
    schema_dir_uri = schema_dir.as_uri() + "/"
    
    # Load schemas
    vlm_schema_path = schema_dir / "vlm_hypothesis.schema.json"
    belief_schema_path = schema_dir / "belief_state.schema.json"
    
    with open(vlm_schema_path, "r", encoding="utf-8") as f:
        vlm_schema = json.load(f)
    
    with open(belief_schema_path, "r", encoding="utf-8") as f:
        belief_schema = json.load(f)
    
    # Build explicit store mapping for $ref resolution
    store = {
        f"{schema_dir_uri}vlm_hypothesis.schema.json": vlm_schema,
        f"{schema_dir_uri}belief_state.schema.json": belief_schema
    }
    
    # Create resolver and validator
    # belief_schema references vlm_schema via $ref, so use belief_schema as referrer
    belief_schema_uri = f"{schema_dir_uri}belief_state.schema.json"
    resolver = RefResolver(base_uri=belief_schema_uri, referrer=belief_schema, store=store)
    vlm_validator = Draft7Validator(vlm_schema, resolver=resolver)
    
    return vlm_validator


def validate_or_error(
    validator: Draft7Validator,
    instance: Any
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Validate instance against schema and return detailed error if invalid.
    
    Args:
        validator: Schema validator instance.
        instance: Instance to validate.
        
    Returns:
        Tuple of (is_valid, error_info). error_info is None if valid, otherwise
        contains keys: message, path (list), schema_path (list), type.
    """
    errors = list(validator.iter_errors(instance))
    if not errors:
        return True, None
    
    # Get first error for details
    error = errors[0]
    return False, {
        "message": error.message,
        "path": list(error.absolute_path),
        "schema_path": list(error.absolute_schema_path),
        "type": "schema"
    }

