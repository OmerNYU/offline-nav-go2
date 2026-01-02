"""Demo helper for node_oracle backend.

This file provides example configurations for the mock perception backend "node_oracle".
The node_oracle backend simulates visibility by mapping specific node IDs to confidence
scores.

USAGE:
------
This is a DEMO file only. To use these examples:

1. Copy the desired map values from DEMO_NODE_ORACLE_MAP below
2. Paste them into perception/config.py, replacing the value of NODE_ORACLE_MAP
3. Run your navigation system

When the robot reaches a node_id that exists in NODE_ORACLE_MAP, perception will
report that the target is visible with the specified confidence level.

CONFIDENCE SCALE:
-----------------
Confidence values range from 0.0 (not visible) to 1.0 (definitely visible).
- 0.90 or higher: High confidence, target clearly visible
- 0.70-0.89: Medium confidence, target likely visible
- 0.50-0.69: Low confidence, target might be visible
- Below 0.50: Very low confidence (typically not used in oracle mode)

EXAMPLE SCENARIOS:
------------------
"""

# Example 1: Single target location (simple demo)
DEMO_NODE_ORACLE_MAP_SIMPLE = {
    12: 0.90,  # Target is visible at node 12 with high confidence
}

# Example 2: Multiple viewpoints (realistic scenario)
DEMO_NODE_ORACLE_MAP_MULTI = {
    12: 0.90,  # Best viewpoint - high confidence
    27: 0.80,  # Alternative viewpoint - good confidence
    35: 0.75,  # Partial view - medium confidence
}

# Example 3: Path-based visibility (target visible along a corridor)
DEMO_NODE_ORACLE_MAP_PATH = {
    10: 0.60,  # Start of corridor - partial view
    11: 0.75,  # Getting closer
    12: 0.90,  # Right in front - best view
    13: 0.75,  # Past it - still visible
    14: 0.60,  # Far end - fading
}

# Default demo map (used in examples below)
DEMO_NODE_ORACLE_MAP = {
    12: 0.90,
    27: 0.80,
}

"""
QUICK START:
------------
To set up a demo where the target becomes visible at nodes 12 and 27:

1. Edit perception/config.py
2. Find the line: NODE_ORACLE_MAP: Dict[int, float] = {}
3. Change it to: NODE_ORACLE_MAP: Dict[int, float] = {12: 0.90, 27: 0.80}
4. Save and run your navigation system

When your robot reaches node 12 or 27, the perception system will report "visible"
with the corresponding confidence level, triggering state transitions in the belief
system (after hysteresis K=2 requirement).

NOTE:
-----
This file is NOT imported automatically. It's purely for reference and documentation.
You must manually copy values to perception/config.py to use them.
"""

