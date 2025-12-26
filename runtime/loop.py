"""Main runtime loop for offline semantic navigation."""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from jsonschema import Draft7Validator, RefResolver

from planner.verifier_stub import VerifierStub
from runtime.logger import DecisionLogger


def load_schemas() -> Tuple[Dict[str, Any], Dict[str, Any], Draft7Validator, Draft7Validator]:
    """Load and set up schema validators with Windows-compatible $ref resolution.
    
    Returns:
        Tuple of (vlm_schema, belief_schema, vlm_validator, belief_validator)
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
    
    # Create resolver and validators
    # belief_schema references vlm_schema via $ref, so use belief_schema as referrer
    belief_schema_uri = f"{schema_dir_uri}belief_state.schema.json"
    resolver = RefResolver(base_uri=belief_schema_uri, referrer=belief_schema, store=store)
    vlm_validator = Draft7Validator(vlm_schema, resolver=resolver)
    belief_validator = Draft7Validator(belief_schema, resolver=resolver)
    
    return vlm_schema, belief_schema, vlm_validator, belief_validator


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
        contains keys: message, path (list), schema_path (list).
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


def generate_mock_vlm_output(rng: random.Random) -> Any:
    """Generate mock VLM output (valid dict, invalid dict, or invalid JSON string).
    
    Args:
        rng: Random number generator for deterministic selection.
        
    Returns:
        Either a valid hypothesis dict, invalid dict, or invalid JSON string.
    """
    roll = rng.random()
    
    # Valid hypotheses (70% probability)
    if roll < 0.7:
        valid_pool = [
            # goto_node action
            {
                "target_status": "visible",
                "action": "goto_node",
                "confidence": 0.9,
                "rationale": "Target detected at node 5",
                "navigation_goal": {
                    "type": "node_id",
                    "node_id": 5
                }
            },
            # ask_clarification action
            {
                "target_status": "ambiguous",
                "action": "ask_clarification",
                "confidence": 0.5,
                "rationale": "Multiple red objects visible",
                "clarification_question": "Which red backpack?"
            },
            # approach action
            {
                "target_status": "visible",
                "action": "approach",
                "confidence": 0.8,
                "rationale": "Target visible, approaching",
                "navigation_goal": {
                    "type": "pose_relative",
                    "distance_meters": 2.0,
                    "angle_degrees": 45.0,
                    "standoff_distance": 0.5
                }
            },
            # explore action (simple, no conditional requirements)
            {
                "target_status": "not_visible",
                "action": "explore",
                "confidence": 0.6,
                "rationale": "Searching for target"
            },
            # rotate action
            {
                "target_status": "not_visible",
                "action": "rotate",
                "confidence": 0.7,
                "rationale": "Scanning environment"
            },
            # stop action
            {
                "target_status": "visible",
                "action": "stop",
                "confidence": 1.0,
                "rationale": "Target reached"
            }
        ]
        return rng.choice(valid_pool)
    
    # Invalid dicts (20% probability)
    elif roll < 0.9:
        invalid_pool = [
            # Missing required fields
            {
                "target_status": "visible",
                "action": "explore"
            },
            # Wrong conditional requirement
            {
                "target_status": "visible",
                "action": "goto_node",
                "confidence": 0.9,
                "rationale": "Target at node"
                # Missing navigation_goal
            },
            # Wrong conditional requirement for approach
            {
                "target_status": "visible",
                "action": "approach",
                "confidence": 0.8,
                "rationale": "Approaching"
                # Missing navigation_goal
            }
        ]
        return rng.choice(invalid_pool)
    
    # Invalid JSON string (10% probability)
    else:
        return "{ this is not json"


def initialize_belief_state(belief_validator: Draft7Validator) -> Dict[str, Any]:
    """Initialize a schema-compliant belief state.
    
    Args:
        belief_validator: Validator for belief state schema.
        
    Returns:
        Initial belief state dictionary.
    """
    belief = {
        "target_status": "searching",  # Valid enum value
        "goal_text": "red backpack",
        "active_constraints": [],
        "candidate_nodes": [],
        "next_action": "explore",  # Valid enum value
        "current_node_id": None,
        "last_vlm_hypothesis": None
    }
    
    # Validate that initial state is schema-compliant
    is_valid, error = validate_or_error(belief_validator, belief)
    if not is_valid:
        raise RuntimeError(f"Initial belief state is invalid: {error}")
    
    return belief


def run_self_check() -> None:
    """Run self-check to validate belief update logic.
    
    Tests:
    - Once target_status becomes "visible", it does NOT revert to "searching" on SKIPPED/verifier failures
    - rejection_reason is set correctly for parse/schema/verifier failures and cleared on ok=True
    - next_action is set to hypothesis.action only on ok=True, else fallback "explore"
    """
    print("Running self-check...")
    
    # Load schemas
    _, _, vlm_validator, belief_validator = load_schemas()
    
    # Helper function to simulate belief update logic (extracted from main loop)
    def update_belief_state(
        belief: Dict[str, Any],
        vlm_validated: Optional[Dict[str, Any]],
        verifier_result: Dict[str, Any],
        meta: Dict[str, Any]
    ) -> None:
        """Simulate the belief update logic from main loop."""
        # 1) last_vlm_hypothesis
        if vlm_validated is not None:
            belief["last_vlm_hypothesis"] = vlm_validated
        else:
            belief["last_vlm_hypothesis"] = None
        
        # 2) target_status: Only PROMOTE to "visible", never auto-demote
        if verifier_result.get("ok", False) and vlm_validated and vlm_validated.get("target_status") == "visible":
            belief["target_status"] = "visible"
        # Otherwise, keep current target_status (no auto-demotion)
        
        # 3) rejection_reason
        if vlm_validated is None:
            if meta.get("validation_error") and meta["validation_error"].get("type") == "json_parse":
                belief["rejection_reason"] = "VLM_INVALID:json_parse"
            else:
                belief["rejection_reason"] = "VLM_INVALID:schema"
        elif not verifier_result.get("ok", False):
            belief["rejection_reason"] = verifier_result.get("reason_code", "UNKNOWN")
        else:
            belief["rejection_reason"] = None
        
        # 4) next_action
        if verifier_result.get("ok", False) and vlm_validated:
            belief["next_action"] = vlm_validated["action"]
        else:
            belief["next_action"] = "explore"
    
    # Test scenario 1: Promote to visible, then verify it doesn't revert on SKIPPED
    belief = initialize_belief_state(belief_validator)
    
    # Step 1: Valid hypothesis with visible -> verifier OK -> should promote to visible
    valid_visible_hypothesis = {
        "target_status": "visible",
        "action": "approach",
        "confidence": 0.9,
        "rationale": "Target visible",
        "navigation_goal": {
            "type": "pose_relative",
            "distance_meters": 2.0,
            "angle_degrees": 45.0,
            "standoff_distance": 0.5
        }
    }
    
    is_valid, _ = validate_or_error(vlm_validator, valid_visible_hypothesis)
    assert is_valid, "Test hypothesis must be valid"
    
    # Simulate verifier OK
    verifier_result_ok = {"ok": True, "reason_code": "OK", "details": {}}
    meta_none = {"validation_error": None}
    update_belief_state(belief, valid_visible_hypothesis, verifier_result_ok, meta_none)
    
    assert belief["target_status"] == "visible", "Should promote to visible on verifier OK + visible hypothesis"
    assert belief["rejection_reason"] is None, "rejection_reason should be None on verifier OK"
    assert belief["next_action"] == "approach", "next_action should be hypothesis.action on verifier OK"
    
    # Step 2: Now simulate a SKIPPED (parse failure)
    # This should NOT demote visible -> searching
    belief_before_status = belief["target_status"]
    verifier_result_skipped = {"ok": False, "reason_code": "SKIPPED", "details": {}}
    meta_parse_error = {"validation_error": {"message": "test", "type": "json_parse"}}
    update_belief_state(belief, None, verifier_result_skipped, meta_parse_error)
    
    assert belief["target_status"] == "visible", \
        f"FAIL: target_status reverted from visible to {belief['target_status']} on SKIPPED"
    assert belief["rejection_reason"] == "VLM_INVALID:json_parse", "rejection_reason should be set on parse failure"
    assert belief["next_action"] == "explore", "next_action should be explore on SKIPPED"
    
    print("[PASS] Test 1: target_status does not auto-demote on SKIPPED")
    
    # Test scenario 2: rejection_reason correctness
    belief = initialize_belief_state(belief_validator)
    
    # Parse failure
    update_belief_state(belief, None, verifier_result_skipped, meta_parse_error)
    assert belief["rejection_reason"] == "VLM_INVALID:json_parse", "Parse failure rejection_reason incorrect"
    
    # Schema failure
    meta_schema_error = {"validation_error": {"message": "test", "path": [], "type": "schema"}}
    update_belief_state(belief, None, verifier_result_skipped, meta_schema_error)
    assert belief["rejection_reason"] == "VLM_INVALID:schema", "Schema failure rejection_reason incorrect"
    
    # Verifier failure
    verifier_result_collision = {"ok": False, "reason_code": "COLLISION_DETECTED", "details": {}}
    update_belief_state(belief, valid_visible_hypothesis, verifier_result_collision, meta_none)
    assert belief["rejection_reason"] == "COLLISION_DETECTED", "Verifier failure rejection_reason incorrect"
    
    # Verifier OK -> should be None
    update_belief_state(belief, valid_visible_hypothesis, verifier_result_ok, meta_none)
    assert belief["rejection_reason"] is None, "Verifier OK should clear rejection_reason"
    
    print("[PASS] Test 2: rejection_reason set correctly")
    
    # Test scenario 3: next_action logic
    belief = initialize_belief_state(belief_validator)
    
    # Verifier OK -> next_action should be hypothesis.action
    valid_hypothesis = {
        "target_status": "not_visible",
        "action": "rotate",
        "confidence": 0.7,
        "rationale": "Scanning"
    }
    is_valid, _ = validate_or_error(vlm_validator, valid_hypothesis)
    assert is_valid, "Test hypothesis must be valid"
    
    update_belief_state(belief, valid_hypothesis, verifier_result_ok, meta_none)
    assert belief["next_action"] == "rotate", "next_action should be hypothesis.action on verifier OK"
    
    # Verifier failed -> next_action should be "explore"
    update_belief_state(belief, valid_hypothesis, verifier_result_collision, meta_none)
    assert belief["next_action"] == "explore", "next_action should be explore on verifier failure"
    
    # SKIPPED -> next_action should be "explore"
    update_belief_state(belief, None, verifier_result_skipped, meta_parse_error)
    assert belief["next_action"] == "explore", "next_action should be explore on SKIPPED"
    
    print("[PASS] Test 3: next_action logic correct")
    
    print("All self-checks passed!")


def main() -> None:
    """Main runtime loop."""
    parser = argparse.ArgumentParser(description="Offline semantic navigation runtime loop")
    parser.add_argument("--steps", type=int, default=20, help="Number of steps to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism")
    parser.add_argument("--self-check", action="store_true", help="Run self-check tests and exit")
    args = parser.parse_args()
    
    if args.self_check:
        run_self_check()
        return
    
    # Load schemas and create validators
    vlm_schema, belief_schema, vlm_validator, belief_validator = load_schemas()
    
    # Initialize components
    rng = random.Random(args.seed)
    verifier = VerifierStub(rng)
    logger = DecisionLogger()
    belief = initialize_belief_state(belief_validator)
    
    # Main loop
    for step_id in range(args.steps):
        # Deep copy belief_before
        belief_before = json.loads(json.dumps(belief))
        
        # Generate mock VLM output
        vlm_raw = generate_mock_vlm_output(rng)
        
        # Initialize meta and verifier_result
        meta: Dict[str, Any] = {
            "vlm_latency_ms": 0.0,
            "verify_latency_ms": 0.0,
            "validation_error": None
        }
        verifier_result: Dict[str, Any] = {
            "ok": False,
            "reason_code": "SKIPPED",
            "details": {}
        }
        
        # Parse and validate VLM output
        vlm_validated: Optional[Dict[str, Any]] = None
        vlm_start = time.perf_counter()
        
        # Try to parse if string
        vlm_dict: Optional[Dict[str, Any]] = None
        if isinstance(vlm_raw, str):
            try:
                vlm_dict = json.loads(vlm_raw)
            except json.JSONDecodeError as e:
                meta["validation_error"] = {
                    "message": str(e),
                    "type": "json_parse"
                }
                # verifier_result already set to SKIPPED above
        else:
            vlm_dict = vlm_raw
        
        # Validate schema if parsing succeeded
        if vlm_dict is not None:
            is_valid, error_info = validate_or_error(vlm_validator, vlm_dict)
            if is_valid:
                vlm_validated = vlm_dict
                meta["validation_error"] = None
            else:
                meta["validation_error"] = error_info
                # verifier_result already set to SKIPPED above
        # else: vlm_dict is None (parse failed), verifier_result already SKIPPED
        
        meta["vlm_latency_ms"] = (time.perf_counter() - vlm_start) * 1000
        
        # Verify hypothesis if valid
        if vlm_validated is not None:
            verify_start = time.perf_counter()
            verifier_result = verifier.verify_hypothesis(vlm_validated)
            meta["verify_latency_ms"] = (time.perf_counter() - verify_start) * 1000
        
        # Determine VLM and planner status for logging
        if vlm_validated is not None:
            vlm_status = "VALID"
        elif meta["validation_error"] and meta["validation_error"].get("type") == "json_parse":
            vlm_status = "PARSE_FAIL"
        else:
            vlm_status = "SCHEMA_BAD"
        
        planner_status = verifier_result.get("reason_code", "UNKNOWN")
        meta["vlm_status"] = vlm_status
        meta["planner_status"] = planner_status
        
        # Update belief state according to architecture rules
        # 1) last_vlm_hypothesis
        if vlm_validated is not None:
            belief["last_vlm_hypothesis"] = vlm_validated
        else:
            belief["last_vlm_hypothesis"] = None
        
        # 2) target_status: Only PROMOTE to "visible", never auto-demote
        if verifier_result.get("ok", False) and vlm_validated and vlm_validated.get("target_status") == "visible":
            belief["target_status"] = "visible"
        # Otherwise, keep current target_status (no auto-demotion)
        
        # 3) rejection_reason
        if vlm_validated is None:
            if meta["validation_error"] and meta["validation_error"].get("type") == "json_parse":
                belief["rejection_reason"] = "VLM_INVALID:json_parse"
            else:
                belief["rejection_reason"] = "VLM_INVALID:schema"
        elif not verifier_result.get("ok", False):
            belief["rejection_reason"] = verifier_result.get("reason_code", "UNKNOWN")
        else:
            belief["rejection_reason"] = None
        
        # 4) next_action
        if verifier_result.get("ok", False) and vlm_validated:
            belief["next_action"] = vlm_validated["action"]
        else:
            belief["next_action"] = "explore"
        
        # Validate updated belief state
        is_valid, error = validate_or_error(belief_validator, belief)
        if not is_valid:
            raise RuntimeError(f"Updated belief state is invalid: {error}")
        
        # Log step
        logger.log_step(
            step_id=step_id,
            belief_before=belief_before,
            vlm_raw=vlm_raw,
            vlm_validated=vlm_validated,
            verifier_result=verifier_result,
            belief_after=belief,
            meta=meta
        )
        
        # Print console summary
        print(f"Step {step_id}: VLM={vlm_status} | Planner={planner_status} | State={belief['target_status']}")


if __name__ == "__main__":
    main()

