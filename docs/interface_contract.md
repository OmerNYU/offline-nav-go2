# Offline Semantic Navigation — Interface Contract v0.1

## Purpose
This document defines the exact interface between:
1) The Vision–Language Model (VLM)
2) The internal Belief State
3) The geometric planner (verification layer)

The system follows the principle:
**VLM proposes — Planner verifies**

---

## A) Belief State (Internal State)

The BeliefState is the single source of truth for decision-making.

Fields:
- target_status: one of ["searching", "visible", "likely_in_memory", "unreachable", "done"]
- goal_text: string (e.g., "red backpack")
- current_node_id: integer or null
- candidate_nodes: list of { node_id: int, score: float }
- active_constraints: list of strings (e.g., ["avoid_carpet"])
- last_vlm_hypothesis: object or null
- rejection_reason: string or null
- next_action: one of ["approach", "explore", "rotate", "goto_node", "ask_clarification", "stop"]

---

## B) VLM Hypothesis (Model Output)

The VLM must output STRICT JSON conforming to `schema/vlm_hypothesis.schema.json`.

Rules:
- The VLM never outputs raw motor commands
- The VLM never assumes reachability
- The VLM may propose goals, search actions, or clarification

---

## C) Verification Rules (Planner Responsibility)

A hypothesis is ACCEPTED only if:
1) A collision-free path exists
2) The goal respects all active constraints
3) The planner explicitly confirms feasibility

If any check fails, the hypothesis is REJECTED.

---

## D) Fallback Policy

On rejection:
- If unreachable → try next candidate node
- If no path → explore or rotate
- If ambiguous → ask clarification
- If repeated failures → stop safely

---

## E) Logged Metrics (Every Cycle)

- hypothesis_action
- accepted / rejected
- rejection_reason
- time_to_hypothesis_ms
- planner_success

