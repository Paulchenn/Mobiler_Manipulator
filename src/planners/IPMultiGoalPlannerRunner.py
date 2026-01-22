import networkx as nx
import numpy as np

class MultiGoalPlannerRunner:
    """
    Executes a path planner sequentially for multiple goals while handling
    Pick & Place actions (defined as tuples in the goalList).
    """

    # Standoff distance configuration from target position (in meters)
    DEFAULT_APPROACH_VEC = [0.6, 0.0]
    # Number of interpolation steps for linear approach/retreat motions
    LINEAR_STEPS = 5
    
    @staticmethod
    def get_standoff_config(target_config, offset_vector):
        """
        Calculates the standoff point based on a local offset vector relative to target orientation.
        
        Args:
            target_config: Target configuration [x, y, theta]
            offset_vector: Local offset [dx, dy] relative to target orientation theta
            
        Returns:
            Standoff configuration with same orientation as target but offset position
        """
        x_target, y_target, theta = target_config[0], target_config[1], target_config[2]

        dx_local, dy_local = offset_vector

        # Transform local offset vector to world frame using target orientation
        # Formula: P_standoff = P_target - R(theta) * V_offset
        # The subtraction places the standoff point "before" approaching the target

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Apply 2D rotation matrix to offset vector
        dx_world = dx_local * cos_t - dy_local * sin_t
        dy_world = dx_local * sin_t + dy_local * cos_t

        # Compute standoff position (target position minus rotated offset)
        standoff_x = x_target - dx_world
        standoff_y = y_target - dy_world
        
        # Copy configuration and update position coordinates
        standoff_config = target_config.copy()
        standoff_config[0] = standoff_x
        standoff_config[1] = standoff_y
        
        return standoff_config
    
    @staticmethod
    def interpolate_linear(start, end, steps):
        """
        Generate linear interpolation trajectory between two configurations.
        
        Args:
            start: Starting configuration
            end: End configuration
            steps: Number of interpolation steps (excluding start, including end)
            
        Returns:
            List of interpolated configurations
        """
        trajectory = []
        s = np.array(start)
        e = np.array(end)
        # Generate interpolation points (start exclusive, end inclusive)
        for i in range(1, steps + 1):
            alpha = i / steps
            point = s * (1 - alpha) + e * alpha
            trajectory.append(point)
        return trajectory

    @staticmethod
    def run_benchmark(planner, benchmark, config):
        """
        Execute multi-goal path planning with Pick & Place operations.
        
        Expected benchmark object attributes:
            - goalList: List of goal configurations. Supports two formats:
                Format A (Recommended): [ (pos, "PICK"), (pos, "PLACE"), (pos, "MOVE") ]
                Format B (Legacy): [ pos, pos, pos ] (implicit MOVE actions)
            - objectShape: Polygon vertices defining object geometry [[x,y], ...]
            
        Returns:
            tuple: (path, action_events, status)
                - path: List of node IDs representing the full trajectory
                - action_events: Dict mapping path indices to (action, object_shape) tuples
                - status: Dict with success flag, failure segment index, and reason
        """
        
        # --- 1. INITIALIZATION & DATA STRUCTURES ---
        full_relabeled_path = []              # Complete trajectory path
        full_roadmap_graph = nx.Graph()      # Combined roadmap from all segments
        full_colliding_edges = []            # Edges causing collisions
        full_non_colliding_edges = []        # Valid collision-free edges

        # Dictionary mapping path indices to (action_type, object_shape) events
        action_events = {}
        
        current_start = benchmark.startList 
        targets = benchmark.goalList

        # Retrieve object shape from collision checker
        object_shape = planner._collisionChecker.get_object_shape()
        
        last_segment_end_node = None
        current_world_obstacle_poly = None

        # Status tracking variables (-1 indicates success)
        failed_segment_index = -1
        failed_reason = ""
        
        # Ensure gripper is empty at start
        if hasattr(planner._collisionChecker, 'detach_object'):
            planner._collisionChecker.detach_object()

        # --- 2. INITIAL OBJECT PLACEMENT ---
        # Place object as static obstacle at first PICK location
        if object_shape is not None:
            for t in targets:
                # Parse goal entry and check for PICK action (tuple format)
                if isinstance(t, (tuple, list)) and len(t) >= 2 and t[1] == "PICK":
                    pick_config = t[0]
                    
                    # Temporarily attach object to compute world frame position
                    planner._collisionChecker.attach_object(object_shape)
                    geo = planner._collisionChecker.get_robot_geometry(pick_config)
                    poly = geo.get('held_object')
                    planner._collisionChecker.detach_object()
                    
                    if poly is not None:
                        # Add object as static obstacle in world frame
                        print(f"        [Setup] Initial object placed at Pick-Location as obstacle.")
                        planner._collisionChecker.obstacles.append(poly)
                        current_world_obstacle_poly = poly
                    break  # Object is placed at first PICK location

        # --- 3. MAIN PLANNING LOOP ---
        for i, target_entry in enumerate(targets):
            
            # Parse goal entry to extract coordinates, action type, and approach offset
            actual_target_coords = None
            current_action = "MOVE"
            current_offset = MultiGoalPlannerRunner.DEFAULT_APPROACH_VEC

            if isinstance(target_entry, (tuple, list)):
                actual_target_coords = target_entry[0]

                # Case: (Config, Action) - length 2
                if len(target_entry) >= 2: current_action = target_entry[1]
                # Case: (Config, Action, Offset) - length 3
                if len(target_entry) >= 3: current_offset = target_entry[2]

            # Case: Config only (legacy format)
            else:
                actual_target_coords = target_entry

            # Determine planning goal: standoff point for PICK/PLACE, direct target for MOVE
            if current_action in ["PICK", "PLACE"]:
                planner_goal_coords = MultiGoalPlannerRunner.get_standoff_config(actual_target_coords, current_offset)
                print(f"        [Runner] Segment {i}: {current_action} with offset {current_offset}. Planning to Standoff.")
            else:
                planner_goal_coords = actual_target_coords
                print(f"        [Runner] Segment {i}: MOVE. Planning directly to target.")

            current_goal_list_for_planner = [planner_goal_coords]

            # Execute path planning from current start to planning goal
            segment = planner.planPath(current_start, current_goal_list_for_planner, config)

            # Error handling: stop execution if no path found
            if segment is None or len(segment) == 0:
                print(f"        [WARNING] No path found in segment {i} ({current_action})! Stopping execution here.")
                failed_segment_index = i
                failed_reason = f"Failed at Segment {i}: {current_action}"
                break  # Abort execution on planning failure
            
            # Merge segment roadmap into full graph with unique node naming
            current_graph = planner.graph
            mapping = {node: f"s{i}_{node}" for node in current_graph.nodes()}
            relabeled_segment = [mapping[n] for n in segment]
            
            relabeled_graph = nx.relabel_nodes(current_graph, mapping)
            full_roadmap_graph = nx.compose(full_roadmap_graph, relabeled_graph)
            
            # Preserve collision information from planner
            if hasattr(planner, 'collidingEdges'):
                for u, v in planner.collidingEdges:
                    if u in mapping and v in mapping:
                        full_colliding_edges.append((mapping[u], mapping[v]))
            if hasattr(planner, 'nonCollidingEdges'):
                for u, v in planner.nonCollidingEdges:
                    if u in mapping and v in mapping:
                        full_non_colliding_edges.append((mapping[u], mapping[v]))

            # Connect previous segment end to current segment start
            if last_segment_end_node is not None:
                full_roadmap_graph.add_edge(last_segment_end_node, relabeled_segment[0], connection="virtual") 

            # Extend full path and track endpoint
            full_relabeled_path.extend(relabeled_segment)
            last_segment_end_node = relabeled_segment[-1]

            # --- ACTION EXECUTION (PICK/PLACE only) ---
            if current_action in ["PICK", "PLACE"]:
                
                # A) Linear approach: Standoff -> Target
                approach_path = MultiGoalPlannerRunner.interpolate_linear(planner_goal_coords, actual_target_coords, MultiGoalPlannerRunner.LINEAR_STEPS)
                
                prev_node_id = last_segment_end_node
                
                # Add approach trajectory nodes to roadmap
                for k, pt in enumerate(approach_path):
                    node_id = f"s{i}_approach_in_{k}"
                    full_roadmap_graph.add_node(node_id, pos=pt)
                    full_roadmap_graph.add_edge(prev_node_id, node_id, connection="linear")
                    full_relabeled_path.append(node_id)
                    prev_node_id = node_id
                
                # Now at target position - record action event index
                current_end_index = len(full_relabeled_path) - 1
                poly_to_activate_after_retreat = None
                
                # Execute PICK action: remove object from world, attach to gripper
                if current_action == "PICK":
                    action_events[current_end_index] = ("PICK", object_shape)
                    
                    if current_world_obstacle_poly in planner._collisionChecker.obstacles:
                        planner._collisionChecker.obstacles.remove(current_world_obstacle_poly)
                        print(f"        [Action] PICK: Removed static object.")
                        current_world_obstacle_poly = None

                    if object_shape is not None:
                        print(f"        [Action] PICK: Grasping.")
                        planner._collisionChecker.attach_object(object_shape)
                
                # Execute PLACE action: release object from gripper, add to world
                elif current_action == "PLACE":
                    action_events[current_end_index] = ("PLACE", None)
                    
                    # Compute object position in world frame at target location
                    geo = planner._collisionChecker.get_robot_geometry(actual_target_coords)
                    placed_poly = geo.get('held_object')
                    
                    print(f"        [Action] PLACE: Releasing.")
                    planner._collisionChecker.detach_object()

                    if placed_poly is not None:
                        print(f"        [Action] PLACE: Object becomes obstacle.")
                        # Activate obstacle after retreat to avoid self-collision
                        poly_to_activate_after_retreat = placed_poly

                # B) Linear retreat: Target -> Standoff
                retreat_path = MultiGoalPlannerRunner.interpolate_linear(actual_target_coords, planner_goal_coords, MultiGoalPlannerRunner.LINEAR_STEPS)
                
                # Add retreat trajectory nodes to roadmap
                for k, pt in enumerate(retreat_path):
                    node_id = f"s{i}_approach_out_{k}"
                    full_roadmap_graph.add_node(node_id, pos=pt)
                    full_roadmap_graph.add_edge(prev_node_id, node_id, connection="linear")
                    full_relabeled_path.append(node_id)
                    prev_node_id = node_id

                # Activate placed object in world frame
                if poly_to_activate_after_retreat is not None:
                    planner._collisionChecker.obstacles.append(poly_to_activate_after_retreat)
                    current_world_obstacle_poly = poly_to_activate_after_retreat
                
                # Robot returns to standoff point for next segment
                current_start = [planner_goal_coords]
                last_segment_end_node = prev_node_id

            else:
                # MOVE action: robot remains at target location for next segment
                current_start = [actual_target_coords]

        # --- CLEANUP & FINALIZATION ---
        # Add marker nodes for start and goal in roadmap
        if full_relabeled_path:
            real_start = full_relabeled_path[0]
            if real_start in full_roadmap_graph.nodes:
                full_roadmap_graph.add_node("start", pos=full_roadmap_graph.nodes[real_start]['pos'])
            real_goal = full_relabeled_path[-1]
            if real_goal in full_roadmap_graph.nodes:
                full_roadmap_graph.add_node("goal", pos=full_roadmap_graph.nodes[real_goal]['pos'])

        # Update planner with complete multi-goal roadmap
        planner.graph = full_roadmap_graph
        planner.collidingEdges = full_colliding_edges         
        planner.nonCollidingEdges = full_non_colliding_edges  
        
        # Clean up gripper and remove any remaining static obstacles
        if hasattr(planner._collisionChecker, 'detach_object'):
            planner._collisionChecker.detach_object()
        if current_world_obstacle_poly in planner._collisionChecker.obstacles:
            planner._collisionChecker.obstacles.remove(current_world_obstacle_poly)

        # Build status dictionary with planning results
        status = {
            "success": (failed_segment_index == -1),
            "fail_segment": failed_segment_index,
            "fail_reason": failed_reason,
            "total_segments": len(targets)
        }
        
        return full_relabeled_path, action_events, status