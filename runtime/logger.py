"""Decision logger for runtime loop steps."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class DecisionLogger:
    """Logger that writes decision steps to JSONL file."""
    
    def __init__(self) -> None:
        """Initialize logger and create logs directory if needed."""
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        self._log_file = logs_dir / "decisions.jsonl"
        self._file = open(self._log_file, "a", encoding="utf-8")
    
    def log_step(
        self,
        step_id: int,
        belief_before: Dict[str, Any],
        vlm_raw: Any,
        vlm_validated: Optional[Dict[str, Any]],
        verifier_result: Dict[str, Any],
        belief_after: Dict[str, Any],
        meta: Dict[str, Any]
    ) -> None:
        """Log a single decision step.
        
        Args:
            step_id: Step identifier.
            belief_before: Belief state before this step.
            vlm_raw: Raw VLM output (string or dict).
            vlm_validated: Validated VLM hypothesis dict or None.
            verifier_result: Verifier result dictionary.
            belief_after: Belief state after this step.
            meta: Metadata dictionary with latencies and validation_error if any.
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "step_id": step_id,
            "event_type": "STEP",
            "belief_before": self._serialize(belief_before),
            "vlm_raw": self._serialize(vlm_raw),
            "vlm_validated": self._serialize(vlm_validated),
            "verifier_result": self._serialize(verifier_result),
            "belief_after": self._serialize(belief_after),
            "meta": self._serialize(meta)
        }
        
        json_line = json.dumps(log_entry, ensure_ascii=False)
        self._file.write(json_line + "\n")
        self._file.flush()
    
    def _serialize(self, obj: Any) -> Any:
        """Serialize object to JSON-serializable format.
        
        Converts Path objects and datetime objects to strings.
        
        Args:
            obj: Object to serialize.
            
        Returns:
            JSON-serializable representation of the object.
        """
        if isinstance(obj, (Path, datetime)):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize(item) for item in obj]
        else:
            return obj
    
    def __del__(self) -> None:
        """Close file on deletion."""
        if hasattr(self, "_file") and self._file:
            self._file.close()

