# Node Oracle Demo Configuration

This document explains how to configure the `node_oracle` backend for demos and testing.

## Overview

The `node_oracle` backend simulates target visibility for testing and demos. You can easily configure which navigation nodes should report the target as visible.

## How It Works

When the robot reaches a node_id in `NODE_ORACLE_MAP`, perception reports the target as visible with the specified confidence (0.0 to 1.0).

## Setup Instructions

1. Edit `perception/config.py`
2. Find: `NODE_ORACLE_MAP: Dict[int, float] = {}`
3. Replace with your map, e.g.:
   ```python
   NODE_ORACLE_MAP: Dict[int, float] = {
       12: 0.90,  # High confidence at node 12
       27: 0.80,  # Good confidence at node 27
   }
   ```
4. Save and run your navigation system

## Example Scenarios

See `perception/demo_oracle.py` for pre-configured examples:

### Single Target Location
```python
DEMO_NODE_ORACLE_MAP_SIMPLE = {
    12: 0.90,  # Target is visible at node 12 with high confidence
}
```

### Multiple Viewpoints
```python
DEMO_NODE_ORACLE_MAP_MULTI = {
    12: 0.90,  # Best viewpoint - high confidence
    27: 0.80,  # Alternative viewpoint - good confidence
    35: 0.75,  # Partial view - medium confidence
}
```

### Path-Based Visibility (Corridor)
```python
DEMO_NODE_ORACLE_MAP_PATH = {
    10: 0.60,  # Start of corridor - partial view
    11: 0.75,  # Getting closer
    12: 0.90,  # Right in front - best view
    13: 0.75,  # Past it - still visible
    14: 0.60,  # Far end - fading
}
```

## Confidence Scale

Confidence values range from 0.0 (not visible) to 1.0 (definitely visible):

- **0.90 or higher**: High confidence, target clearly visible
- **0.70-0.89**: Medium confidence, target likely visible
- **0.50-0.69**: Low confidence, target might be visible
- **Below 0.50**: Very low confidence (typically not used in oracle mode)

## State Transitions

**Important:** After visibility is detected, the system requires K=2 consecutive hits (hysteresis) before the belief system transitions to "visible" state. This prevents flapping from false positives.

For full details on the perception visibility gate system, see `perception_visibility_gate.md`.

## Project Structure

```
├── perception/        # Perception and visibility checking
│   ├── config.py      # Configuration (edit NODE_ORACLE_MAP here)
│   ├── check_visibility.py
│   └── demo_oracle.py # Demo examples (reference only)
├── vlm/               # Vision-language model integration
├── memory/            # Memory storage and retrieval
├── planner/           # Planning and verification
├── runtime/           # Main execution loop
├── schema/            # JSON schemas for validation
├── tests/             # Test suite
└── docs/              # Documentation
```

## Testing Your Configuration

Run the full test suite to verify your setup:
```bash
pytest
```

Run specific perception tests:
```bash
pytest tests/test_perception_integration.py
pytest tests/test_perception_check_visibility.py
```

## Related Documentation

- **Perception Visibility Gate**: `perception_visibility_gate.md` - Complete visibility detection system
- **Memory-Guided Search**: `memory_guided_search_implementation.md` - Semantic memory integration
- **VLM Integration**: `ollama_vlm_integration.md` - Vision-language model setup
- **Interface Contract**: `interface_contract.md` - System interfaces

