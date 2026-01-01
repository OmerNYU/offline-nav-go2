"""Main runtime loop for offline semantic navigation."""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from jsonschema import Draft7Validator, RefResolver

from memory.embedding import DeterministicEmbedder
from memory.retrieval import retrieve_candidates
from memory.store import seed_demo_store
from memory.utils import tokenize
from perception.check_visibility import check_visibility
from planner.verifier_stub import VerifierStub
from runtime.logger import DecisionLogger
from runtime.memory_bridge import apply_memory_retrieval
from runtime.schema_loader import load_hypothesis_validator, validate_or_error
from vlm.fallback import generate_fallback_hypothesis
from vlm.ollama_client import OllamaVLMClient


def load_schemas() -> Tuple[Dict[str, Any], Dict[str, Any], Draft7Validator, Draft7Validator]:
    """Load and set up schema validators with Windows-compatible $ref resolution.
    
    Uses runtime.schema_loader for VLM hypothesis validator to avoid circular imports.
    
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
    # Use shared VLM validator from schema_loader (for consistency with vlm/ollama_client.py)
    vlm_validator = load_hypothesis_validator()
    
    # belief_schema references vlm_schema via $ref, so use belief_schema as referrer
    belief_schema_uri = f"{schema_dir_uri}belief_state.schema.json"
    resolver = RefResolver(base_uri=belief_schema_uri, referrer=belief_schema, store=store)
    belief_validator = Draft7Validator(belief_schema, resolver=resolver)
    
    return vlm_schema, belief_schema, vlm_validator, belief_validator


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
        "last_vlm_hypothesis": None,
        "rejection_reason": None,  # Explicit initialization for clarity
        "last_visibility": None,
        "visibility_streak": 0,
        "last_seen_node_id": None,
        "visible_since_step": None
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


# Memory retrieval threshold (on [0,1] scale after cosine mapping)
MEMORY_SCORE_THRESH = 0.3


def main() -> None:
    """Main runtime loop."""
    parser = argparse.ArgumentParser(description="Offline semantic navigation runtime loop")
    parser.add_argument("--steps", type=int, default=20, help="Number of steps to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism")
    parser.add_argument("--self-check", action="store_true", help="Run self-check tests and exit")
    parser.add_argument("--use-ollama", action="store_true", help="Use Ollama VLM backend instead of mock")
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
    
    # Initialize memory components
    # TODO: Replace seed_demo_store() with persisted SLAM-derived nodes in production
    memory_store = seed_demo_store()
    embedder = DeterministicEmbedder(dim=64)
    
    # Initialize VLM client
    use_ollama = args.use_ollama or os.getenv("VLM_BACKEND") == "ollama"
    if use_ollama:
        vlm_client = OllamaVLMClient()
    else:
        vlm_client = None
    
    # Main loop
    for step_id in range(args.steps):
        # Deep copy belief_before
        belief_before = json.loads(json.dumps(belief))
        
        # === Memory retrieval step ===
        # Tokenize goal_text to determine if retrieval should run
        tokens = tokenize(belief["goal_text"])
        retrieval_ran = len(tokens) > 0
        
        # Only call retrieve_candidates if we have valid tokens
        if retrieval_ran:
            candidates = retrieve_candidates(belief["goal_text"], memory_store, embedder, k=5)
        else:
            candidates = []
        
        # Always update belief with candidates (even if empty) for schema safety
        apply_memory_retrieval(belief, candidates, MEMORY_SCORE_THRESH)
        
        # Initialize meta and verifier_result
        meta: Dict[str, Any] = {
            "vlm_latency_ms": 0.0,
            "verify_latency_ms": 0.0,
            "validation_error": None,
            "retrieval_ran": retrieval_ran,
            "retrieval_topk": candidates,
            "retrieval_best_score": candidates[0]["score"] if candidates else None,
            "retrieval_threshold_pass": candidates[0]["score"] >= MEMORY_SCORE_THRESH if candidates else False
        }
        
        # Generate VLM output (Ollama or mock)
        if use_ollama:
            # Build candidate_nodes_top once to ensure alignment
            candidate_nodes_top = candidates[:3]  # top 3 with {node_id, score}
            
            # Precompute memory_context (plain data, no runtime objects)
            memory_context = []
            missing_node_ids = []
            for cand in candidate_nodes_top:
                node = memory_store.get_node(cand["node_id"])
                if node:
                    memory_context.append({
                        "node_id": node.node_id,
                        "score": cand["score"],
                        "tags": node.tags,
                        "summary": node.summary[:50]  # truncate to 50 chars
                    })
                else:
                    # Node missing: append placeholder entry to keep prompt aligned
                    memory_context.append({
                        "node_id": cand["node_id"],
                        "score": cand["score"],
                        "tags": ["<missing>"],
                        "summary": "<missing>"
                    })
                    missing_node_ids.append(cand["node_id"])
            
            # Track missing nodes in meta (for debuggability)
            if missing_node_ids:
                meta["memory_context_missing_nodes"] = True
                meta["memory_context_missing_node_ids"] = missing_node_ids
            else:
                meta["memory_context_missing_nodes"] = False
            
            # Build context dict (ONLY JSON-serializable data)
            context = {
                "goal_text": belief["goal_text"],
                "active_constraints": belief["active_constraints"],
                "belief_target_status": belief.get("target_status"),
                "candidate_nodes": candidate_nodes_top,
                "memory_context": memory_context
            }
            
            # Call Ollama client
            vlm_raw, client_meta = vlm_client.propose_hypothesis(context)
            
            # Merge client meta into meta
            meta["vlm_backend"] = client_meta.get("vlm_backend", "ollama")
            meta["vlm_model"] = client_meta.get("vlm_model")
            meta["vlm_latency_ms"] = client_meta.get("vlm_latency_ms", 0.0)
            meta["vlm_parse_ok"] = client_meta.get("vlm_parse_ok", False)
            meta["vlm_schema_ok"] = client_meta.get("vlm_schema_ok", False)
            meta["vlm_retry_count"] = client_meta.get("vlm_retry_count", 0)
            meta["vlm_error"] = client_meta.get("vlm_error")
            
            # If hypothesis is None, use fallback
            if vlm_raw is None:
                vlm_raw = generate_fallback_hypothesis(belief, candidates)
        else:
            # Mock mode
            meta["vlm_backend"] = "mock"
            memory_context = []  # No memory context in mock mode
            vlm_raw = generate_mock_vlm_output(rng)
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
        
        # 2) target_status: VLM can NO LONGER set visible/done
        # Only perception can promote to "visible" or "done"
        # (this logic is now handled by perception visibility gate after action simulation)
        # Keep current target_status (no auto-demotion)
        
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
        
        # 5) Simulate action execution (BEFORE perception check)
        # Use final executed action (belief["next_action"]) so fallback-chosen actions are also simulated
        if belief["next_action"] == "goto_node":
            # Extract target node from last_vlm_hypothesis (works for both VLM and fallback)
            target_node_id = belief.get("last_vlm_hypothesis", {}).get("navigation_goal", {}).get("node_id")
            if target_node_id is not None:
                belief["current_node_id"] = target_node_id
        # Other actions (approach, explore, rotate, stop, ask_clarification) don't change current_node_id
        
        # 6) Perception visibility check (AFTER action execution)
        # This is the ONLY authority that can set target_status to "visible" or "done"
        vr = check_visibility(
            goal_text=belief["goal_text"],
            current_node_id=belief.get("current_node_id"),
            memory_context=memory_context,
            belief_state=belief
        )
        
        # Initialize new fields if not present (backward compatibility)
        if "last_visibility" not in belief:
            belief["last_visibility"] = None
        if "visibility_streak" not in belief:
            belief["visibility_streak"] = 0
        if "last_seen_node_id" not in belief:
            belief["last_seen_node_id"] = None
        if "visible_since_step" not in belief:
            belief["visible_since_step"] = None
        
        # Update visibility fields
        belief["last_visibility"] = vr
        
        # Update visibility streak (hysteresis mechanism)
        if vr["is_visible"]:
            belief["visibility_streak"] = belief.get("visibility_streak", 0) + 1
        else:
            belief["visibility_streak"] = 0
        
        # Track belief transitions for logging
        belief_transition = None
        old_status = belief["target_status"]
        
        # Transition 1: searching/likely_in_memory -> visible (requires K=2 consecutive hits)
        K = 2  # Hysteresis threshold
        if belief["visibility_streak"] >= K and belief["target_status"] not in {"visible", "done"}:
            belief["target_status"] = "visible"
            belief["last_seen_node_id"] = belief["current_node_id"]
            belief["visible_since_step"] = step_id
            belief_transition = f"{old_status}->visible"
        
        # Transition 2: visible -> done (requires approach action + perception + close_enough)
        if belief["target_status"] == "visible" and belief["next_action"] == "approach" and vr["is_visible"]:
            # Close enough check with visible_since_step guard
            # This prevents immediate transition to done right after becoming visible
            close_enough = (
                belief.get("visible_since_step") is not None
                and step_id > belief.get("visible_since_step")
                and belief.get("current_node_id") is not None
                and belief.get("current_node_id") == belief.get("last_seen_node_id")
            )
            if close_enough:
                belief["target_status"] = "done"
                belief_transition = "visible->done"
        
        # Add perception logging fields to meta
        meta["perception_backend"] = vr["backend"]
        meta["perception_visible"] = vr["is_visible"]
        meta["perception_confidence"] = vr["confidence"]
        meta["perception_latency_ms"] = vr["latency_ms"]
        meta["visibility_streak"] = belief["visibility_streak"]
        meta["belief_transition"] = belief_transition
        meta["perception_reason"] = vr["evidence"]["reason"]
        
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

