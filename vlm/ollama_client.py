"""Ollama VLM client for hypothesis generation.

Calls local Ollama API (qwen2.5vl:7b) to generate VLMHypothesis JSON.
Schema-first validation with robust error handling and safe fallback.
"""

import json
import re
import time
import urllib.request
import urllib.error
from typing import Any, Dict, Optional, Tuple

from runtime.schema_loader import load_hypothesis_validator, validate_or_error


class OllamaVLMClient:
    """Client for generating VLM hypotheses using Ollama API."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5vl:7b",
        timeout_s: float = 30.0,
        max_retries: int = 1
    ):
        """Initialize Ollama VLM client.
        
        Args:
            base_url: Ollama API base URL
            model: Model name to use
            timeout_s: Request timeout in seconds
            max_retries: Maximum number of retries on invalid output
        """
        self.base_url = base_url
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        
        # Load schema validator (Windows-safe, reuses same pattern as loop.py)
        self.validator = load_hypothesis_validator()
    
    def propose_hypothesis(
        self,
        context: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Generate VLM hypothesis from context.
        
        Args:
            context: Plain data dict with keys:
                - goal_text: str
                - active_constraints: list[str]
                - belief_target_status: str | None
                - candidate_nodes: list[dict] with {node_id, score}
                - memory_context: list[dict] with {node_id, score, tags, summary}
        
        Returns:
            Tuple of (hypothesis_dict | None, meta_dict).
            hypothesis_dict is None if generation failed or output invalid.
            meta_dict contains: vlm_backend, vlm_model, vlm_latency_ms,
            vlm_parse_ok, vlm_schema_ok, vlm_retry_count, vlm_error
        """
        start_time = time.perf_counter()
        
        meta: Dict[str, Any] = {
            "vlm_backend": "ollama",
            "vlm_model": self.model,
            "vlm_latency_ms": 0.0,
            "vlm_parse_ok": False,
            "vlm_schema_ok": False,
            "vlm_retry_count": 0,
            "vlm_error": None
        }
        
        # Build prompt
        prompt = self._build_prompt(context)
        
        # Try generation with retry
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                meta["vlm_retry_count"] = attempt
                # Build repair prompt for retry
                prompt = self._build_repair_prompt()
            
            # Call Ollama API
            response_text, error = self._call_ollama_api(prompt)
            
            if error:
                meta["vlm_error"] = error
                meta["vlm_latency_ms"] = (time.perf_counter() - start_time) * 1000
                return None, meta
            
            # Extract JSON from response
            hypothesis_dict = self._extract_json(response_text)
            
            if hypothesis_dict is None:
                meta["vlm_parse_ok"] = False
                if attempt < self.max_retries:
                    continue  # Retry
                else:
                    meta["vlm_error"] = "json_extraction_failed"
                    meta["vlm_latency_ms"] = (time.perf_counter() - start_time) * 1000
                    return None, meta
            
            meta["vlm_parse_ok"] = True
            
            # Validate against schema
            is_valid, error_info = validate_or_error(self.validator, hypothesis_dict)
            
            if not is_valid:
                meta["vlm_schema_ok"] = False
                if attempt < self.max_retries:
                    continue  # Retry
                else:
                    meta["vlm_error"] = f"schema_invalid:{error_info.get('message', 'unknown')}"
                    meta["vlm_latency_ms"] = (time.perf_counter() - start_time) * 1000
                    return None, meta
            
            # Success!
            meta["vlm_schema_ok"] = True
            meta["vlm_latency_ms"] = (time.perf_counter() - start_time) * 1000
            return hypothesis_dict, meta
        
        # Should not reach here, but handle edge case
        meta["vlm_error"] = "max_retries_exceeded"
        meta["vlm_latency_ms"] = (time.perf_counter() - start_time) * 1000
        return None, meta
    
    def _call_ollama_api(self, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        """Call Ollama /api/generate endpoint.
        
        Args:
            prompt: Prompt text to send
        
        Returns:
            Tuple of (response_text | None, error_msg | None).
            If successful, returns (response_text, None).
            If failed, returns (None, error_msg).
        """
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        }
        
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                response_data = json.loads(response.read().decode("utf-8"))
                
                # Handle Ollama error payloads explicitly
                if "error" in response_data:
                    error_msg = response_data["error"]
                    return None, f"ollama_error:{error_msg}"
                
                # Extract response field
                if "response" not in response_data:
                    return None, "missing_response_field"
                
                return response_data["response"], None
        
        except urllib.error.URLError as e:
            return None, f"connection_failed:{str(e)}"
        except urllib.error.HTTPError as e:
            return None, f"http_error:{e.code}"
        except TimeoutError:
            return None, "timeout"
        except json.JSONDecodeError as e:
            return None, f"response_parse_error:{str(e)}"
        except Exception as e:
            return None, f"unexpected_error:{str(e)}"
    
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON dict from text using robust strategies.
        
        Strategies (in order):
        a) Check for JSON inside ```json ... ``` code fences
        b) Balanced-brace scan (string-safe)
        c) Regex patterns as last resort
        
        Safety rule: Only return dict if json.loads() succeeds.
        
        Args:
            text: Text potentially containing JSON
        
        Returns:
            Parsed dict if successful, None otherwise
        """
        if not text:
            return None
        
        # Strategy a: Check for code fences
        code_fence_pattern = r"```json\s*\n(.*?)\n```"
        match = re.search(code_fence_pattern, text, re.DOTALL)
        if match:
            json_text = match.group(1).strip()
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                pass  # Fall through to other strategies
        
        # Strategy b: Balanced-brace scan (string-safe)
        result = self._extract_json_balanced_braces(text)
        if result is not None:
            return result
        
        # Strategy c: Regex patterns as last resort
        # Try to find JSON-like pattern
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.finditer(json_pattern, text)
        for match in matches:
            json_text = match.group(0)
            try:
                result = json.loads(json_text)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                continue
        
        return None
    
    def _extract_json_balanced_braces(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract first complete JSON object using balanced-brace scan.
        
        String-safe: tracks in_string and escape to handle braces inside strings.
        
        Args:
            text: Text containing JSON
        
        Returns:
            Parsed dict if successful, None otherwise
        """
        # Find first opening brace
        start_idx = text.find('{')
        if start_idx == -1:
            return None
        
        # Scan forward with string-safe brace counting
        brace_count = 0
        in_string = False
        escape = False
        
        for i in range(start_idx, len(text)):
            char = text[i]
            
            if escape:
                escape = False
                continue
            
            if char == '\\':
                escape = True
                continue
            
            if char == '"':
                in_string = not in_string
                continue
            
            # Only count braces when not in string
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    
                    if brace_count == 0:
                        # Found matching closing brace
                        json_text = text[start_idx:i+1]
                        try:
                            result = json.loads(json_text)
                            if isinstance(result, dict):
                                return result
                        except json.JSONDecodeError:
                            pass
                        break
        
        return None
    
    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt from context.
        
        Args:
            context: Context dict with goal_text, active_constraints,
                    belief_target_status, candidate_nodes, memory_context
        
        Returns:
            Formatted prompt string
        """
        goal_text = context.get("goal_text", "")
        active_constraints = context.get("active_constraints", [])
        belief_target_status = context.get("belief_target_status", "unknown")
        memory_context = context.get("memory_context", [])
        
        # Format constraints
        constraints_str = ", ".join(active_constraints) if active_constraints else "none"
        
        # Format memory nodes
        memory_str = ""
        for node_ctx in memory_context:
            node_id = node_ctx.get("node_id", "?")
            score = node_ctx.get("score", 0.0)
            tags = node_ctx.get("tags", [])
            summary = node_ctx.get("summary", "")
            tags_str = ", ".join(tags) if tags else "none"
            memory_str += f"  - node_id={node_id}, score={score:.2f}, tags=[{tags_str}], summary=\"{summary}\"\n"
        
        if not memory_str:
            memory_str = "  (no memory nodes available)\n"
        
        prompt = f"""You are a navigation assistant. Output ONLY valid JSON, no other text.

Required schema:
- target_status: one of ["visible", "not_visible", "ambiguous"] (OUTPUT field - different from belief status)
- action: one of ["approach", "explore", "rotate", "goto_node", "ask_clarification", "stop"]
- confidence: number in [0.0, 1.0]
- rationale: string (max 240 chars)

Conditional requirements:
- If action="goto_node": must include navigation_goal with type="node_id" and node_id (integer)
- If action="approach": must include navigation_goal with type="pose_relative", distance_meters, angle_degrees, standoff_distance
- If action="ask_clarification": must include clarification_question (string, max 160 chars)

Current context:
- Goal: {goal_text}
- Constraints: {constraints_str}
- Belief status: {belief_target_status} (for reference only - OUTPUT target_status must be ONLY one of ["visible","not_visible","ambiguous"])
- Candidate memory nodes (top 3):
{memory_str}
Output your hypothesis as JSON:"""
        
        return prompt
    
    def _build_repair_prompt(self) -> str:
        """Build repair prompt for retry after invalid output.
        
        Returns:
            Repair prompt string
        """
        return """You returned invalid JSON. Output ONLY valid JSON with no extra text.

Required schema:
- target_status: one of ["visible", "not_visible", "ambiguous"]
- action: one of ["approach", "explore", "rotate", "goto_node", "ask_clarification", "stop"]
- confidence: number in [0.0, 1.0]
- rationale: string (max 240 chars)

Conditional requirements:
- If action="goto_node": must include navigation_goal with type="node_id" and node_id (integer)
- If action="approach": must include navigation_goal with type="pose_relative", distance_meters, angle_degrees, standoff_distance
- If action="ask_clarification": must include clarification_question (string, max 160 chars)

Output valid JSON now:"""

