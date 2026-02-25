# Smart Forager Example

This example demonstrates the advanced environmental perception capabilities of `mesa-llm`, including:
- **Object Perception**: Detecting non-agent entities (Fruit, Boulders).
- **Navigability**: Understanding which cells are blocked by obstacles.
- **Bearing/Direction**: Getting precise directional information to targets.

## Scenario
An agent with limited energy must find and eat **Fruit** to survive, while navigating around **Boulders** that block movement.

## Key Perception Features in this Example

### 1. Object Detection
The agent is configured with `perceive_objects="all"`. In its observation, it will see:
```json
"objects": [{"type": "fruit", "pickable": True, "nutrition": 10}]
```

### 2. Navigability
If a cell contains a **Boulder**, its observation will automatically report:
```json
"navigable": False
```
The LLM uses this to choose clear paths.

### 3. Spatial Metadata
Every visible cell includes `direction` (e.g., "north") and `relative_bearing` (e.g., `0.0`), allowing the LLM to provide precise movement instructions.

## Running the Example
1. Set your API key:
   ```bash
   export MESA_LLM_API_KEY="your-key-here"
   ```
2. Run the application:
   ```bash
   python examples/smart_forager/app.py
   ```

## Requirements
- `mesa`
- `mesa-llm`
