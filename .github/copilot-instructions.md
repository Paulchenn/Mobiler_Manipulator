# Copilot Instructions: Mobile Manipulator Path Planning

## Project Overview

This is a **5-DOF planar mobile manipulator** system for collision checking and motion planning research. The system combines a mobile base (3-DOF: x, y, θ) with a 2-DOF robotic arm.

**Key focus**: Custom collision detection engine + sampling-based path planners (LazyPRM, VisibilityPRM, RRT, A*) evaluated via automated benchmarking.

## Architecture & Data Flow

### Core Components

1. **`src/collision_checker.py`** - Collision Detection Engine
   - `CollisionChecker` class: Core geometric collision detection using Shapely (polygons, line segments)
   - Handles 3 collision types: base-obstacle, arm-obstacle, arm-self-collision
   - Provides forward kinematics and configuration space validation
   - Supports Pick & Place via dynamic object attachment/detachment

2. **`src/planners/`** - Motion Planning Algorithms
   - **Base class hierarchy**: `PlanerBase` (validates start/goal) → `PRMBase` → specific implementations
   - **PRM variants**: `IPBasicPRM.py`, `IPLazyPRM.py`, `IPVisibilityPRM.py` (standard + multi-goal)
   - **Tree-based**: `IPRRT.py` (RRT), `IPAStar.py` (grid-based search)
   - All planners implement `planPath(startList, goalList, config)` interface
   - All use NetworkX graphs (`self.graph`) to represent roadmaps/trees

3. **Evaluation Pipeline**
   - `IPBenchmark.py`: Defines test scenarios (robot config, obstacles, start/goal sequences)
   - `IPBatchEvaluator.py`: Executes multiple runs (N=10+), collects success/timing/path-length metrics
   - `IPResultCollection.py`: Aggregates results with performance data
   - `IPMultiGoalPlannerRunner.py`: Handles multi-goal sequences with Pick & Place actions

4. **Visualization**
   - `IPAnimator.py`: Real-time plotting of robot configuration and paths
   - `IPSingleRunPlotter.py`, `IPBenchmarkPlotter.py`: Post-execution visualization

## Configuration Space Representation

```python
config = [x, y, theta, q1, q2]  # 5-DOF
# x, y: Mobile base position (meters)
# theta: Base orientation (radians)
# q1, q2: Joint angles (radians)
```

Robot geometry defined in `IPTestSuite.py`:
- `ROBOT_BASE_SHAPE`: Polygon vertices of mobile base
- `ROBOT_ARM_CONFIG`: List of [length, width, [min, max]] per joint
- `GRIPPER_SHAPE`, `PICK_OBJECT`: Polygon definitions for manipulation

## Key Workflows

### Running the Main Notebook
- Entry point: `notebooks/Mobile_Manipulator_Main.ipynb`
- Uses ipywidgets for interactive control (sliders, dropdowns)
- Autoreload enabled for hot-reloading `src/` changes without kernel restart
- Path setup crucial: Explicitly adds `src/`, `src/planners/` to Python path

### Creating a Collision Checker
```python
from collision_checker import CollisionChecker
cc = CollisionChecker(
    base_shape=ROBOT_BASE_SHAPE,
    arm_config=ROBOT_ARM_CONFIG,
    gripper_config=GRIPPER_SHAPE,
    gripper_len=0.1,
    arm_base_offset=ARM_OFFSET,
    limits=LIMITS,
    check_self_collision_flag=True
)
cc.set_obstacles(obstacles)
```

### Adding a New Planner
1. Inherit from `PRMBase` or `PlanerBase`
2. Implement `planPath(startList, goalList, config)` with `@IPPerfMonitor` decorator
3. Use NetworkX graph: `self.graph` (nodes have `'pos'` attribute)
4. Call `self._checkStartGoal()` to validate configurations
5. Use `self._collisionChecker.pointInCollision()`, `lineInCollision()` for queries
6. Return path as node sequence

### Benchmarking Workflow
```python
from IPBatchEvaluator import BatchEvaluator
from IPTestSuite import benchList

planner_factory = {
    "LazyPRM": (IPLazyPRM, config_dict, metadata),
    "VisibilityPRM": (IPVisibilityPRM, config_dict, metadata)
}
results_df = BatchEvaluator.run_experiment(planner_factory, benchList, num_runs=10)
```

## Important Conventions

- **Planner instantiation**: Fresh instance per run (stateless design for reproducibility)
- **Performance monitoring**: Use `@IPPerfMonitor` decorator to auto-collect timing/node counts
- **Pick & Place actions**: Defined in goals as tuples: `(config, action_type, retract_vector)`
  - Actions: `"PICK"`, `"PLACE"`, `"MOVE"`
- **Self-collision flag**: `SELF_CHECK = True/False` in `IPTestSuite.py` controls arm-base collision
- **Virtual environment**: Activate via `source .venv/bin/activate` (macOS/Linux) before running

## Cross-Component Dependencies

- **Planners depend on CollisionChecker**: Pass instance to planner constructor
- **Benchmarks wrap CollisionChecker**: Each benchmark has its own checker instance with obstacle set
- **Batch Evaluator creates fresh planners**: Ensures no state leakage between runs
- **IPTestSuite is the source of truth**: Robot geometry, obstacles, start/goal sequences defined once, reused across all benchmarks

## Testing & Debugging

- Run single benchmarks via notebook cells for interactive debugging
- Check `runs/` directory for saved experiment results (timestamped folders)
- Use `IPAnimator` to visualize collision checker behavior
- `IPPerfMonitor` decorator provides execution profiling (available in result collections)
