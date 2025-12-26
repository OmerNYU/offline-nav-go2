"""Planner verifier stub for hypothesis verification."""

import random
from typing import Dict


class VerifierStub:
    """Stub verifier that simulates planner verification outcomes."""
    
    def __init__(self, rng: random.Random) -> None:
        """Initialize verifier with a random number generator for determinism.
        
        Args:
            rng: Random number generator instance for deterministic outcomes.
        """
        self._rng = rng
    
    def verify_hypothesis(self, hypothesis: Dict) -> Dict:
        """Verify a VLM hypothesis against geometric constraints.
        
        Simulates planner verification with deterministic outcomes:
        - 80%: OK
        - 10%: COLLISION_DETECTED
        - 10%: CONSTRAINT_VIOLATION
        
        Args:
            hypothesis: Validated VLM hypothesis dictionary.
            
        Returns:
            Dictionary with keys:
                - ok: bool indicating if verification passed
                - reason_code: str indicating outcome (OK, COLLISION_DETECTED, CONSTRAINT_VIOLATION)
                - details: dict with additional information (never empty)
        """
        roll = self._rng.random()
        
        if roll < 0.8:
            return {
                "ok": True,
                "reason_code": "OK",
                "details": {"note": "Path verified successfully"}
            }
        elif roll < 0.9:
            return {
                "ok": False,
                "reason_code": "COLLISION_DETECTED",
                "details": {"note": "Collision detected in planned path"}
            }
        else:
            return {
                "ok": False,
                "reason_code": "CONSTRAINT_VIOLATION",
                "details": {"note": "Path violates active constraints"}
            }

