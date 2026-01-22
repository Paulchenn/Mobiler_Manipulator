# src/planners/IPAnimator.py

import datetime
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, Animation
from IPython.display import HTML, display, clear_output
from ipywidgets import Button, Dropdown, VBox, HBox, Output, Layout, Label, IntProgress

# Note: Depending on the specific CollisionChecker implementation, 
# you might need to import it for type hinting, e.g.:
# from src.planners.CollisionChecker import CollisionChecker

class IPAnimator:
    """
    Helper class to generate smooth animations for Robot Path Planning results.
    
    This class handles the visualization of robot trajectories, including 
    complex Pick & Place sequences, by interpolating configurations and 
    managing the state of held objects during the animation.
    """

    @staticmethod
    def _interpolate_line(
        start_pos: List[float] or np.ndarray, 
        end_pos: List[float] or np.ndarray, 
        step_size: float = 0.2
    ) -> List[np.ndarray]:
        """
        Linearly interpolates between two configuration points.

        Args:
            start_pos (list or np.array): The starting configuration.
            end_pos (list or np.array): The target configuration.
            step_size (float): The maximum distance between interpolated points.

        Returns:
            List[np.ndarray]: A list of intermediate configurations including the start.
        """
        start = np.array(start_pos)
        end = np.array(end_pos)
        dist = np.linalg.norm(end - start)
        
        # Handle identical points
        if dist < 1e-6:
            return [start]
        
        # Calculate number of steps required
        n_steps = int(dist / step_size)
        if n_steps < 1: 
            n_steps = 1
        
        steps = []
        for i in range(n_steps):
            alpha = i / n_steps
            interp = start * (1 - alpha) + end * alpha
            steps.append(interp)
        
        return steps

    @staticmethod
    def _get_trajectory_with_state(
        config_path: List[Any], 
        actions: Dict[int, Tuple[str, Any]], 
        cc: Any, 
        step_size: float = 0.2
    ) -> List[Tuple[np.ndarray, Any, Any]]:
        """
        Generates a smooth trajectory containing the configuration and object state for every frame.
        
        This method simulates the robot's movement and gripper state (Pick/Place) 
        step-by-step to ensure the animation correctly renders when an object is 
        picked up or placed down.

        Args:
            config_path (list): List of configurations (or nodes) from the planner.
            actions (dict): Dictionary mapping node indices to actions. 
                            Format: { index: ("ACTION_TYPE", object_shape) }.
            cc (CollisionChecker): Instance of the collision checker to calculate geometry.
            step_size (float): Interpolation step size.

        Returns:
            list: A list of tuples, where each tuple represents a frame:
                  (robot_configuration, held_object_shape, world_object_shape).
        """
        trajectory = []
        
        # --- 1. Determine Initial Object Position ---
        # Logic: If there is a PICK action later, the object must exist in the world 
        # at the start. We simulate the grab briefly to calculate its world position.
        current_world_obj_poly = None
        current_held_object = None  # Robot carries nothing at start

        first_pick_index = -1
        object_shape_def = None

        if actions:
            # Find the first "PICK" action
            for idx, (act, shape) in actions.items():
                if act == "PICK":
                    first_pick_index = idx
                    object_shape_def = shape
                    break

        if first_pick_index != -1 and first_pick_index < len(config_path):
            # Temporarily simulate object being grasped to retrieve its start coordinates
            pick_config = config_path[first_pick_index]
            
            cc.attach_object(object_shape_def)
            geo = cc.get_robot_geometry(pick_config)
            
            if geo.get('held_object') is not None:
                current_world_obj_poly = geo['held_object']  # Polygon in world coordinates
            
            cc.detach_object()
        
        # --- 2. Interpolate Path and Track State ---
        for i in range(len(config_path) - 1):
            start_conf = config_path[i]
            end_conf = config_path[i+1]
            
            # Interpolate movement between two path nodes
            segment_configs = IPAnimator._interpolate_line(start_conf, end_conf, step_size)
            
            # Append interpolated frames with the CURRENT state
            for conf in segment_configs:
                trajectory.append((conf, current_held_object, current_world_obj_poly))
            
            # --- 3. Handle State Transitions (Pick/Place) ---
            target_node_index = i + 1
            if actions and target_node_index in actions:
                action_type, obj_shape = actions[target_node_index]
                target_config = config_path[target_node_index]
                
                if action_type == "PICK":
                    # Transition: Object moves from World -> Robot Hand
                    current_held_object = obj_shape
                    current_world_obj_poly = None

                elif action_type == "PLACE":
                    # Transition: Object moves from Robot Hand -> World
                    # We must attach briefly to calculate where the object lands
                    cc.attach_object(current_held_object)
                    geo = cc.get_robot_geometry(target_config)
                    current_world_obj_poly = geo['held_object']
                    cc.detach_object()
                    
                    current_held_object = None

        # Append the final configuration frame
        trajectory.append((config_path[-1], current_held_object, current_world_obj_poly))
        
        return trajectory

    @staticmethod
    def animate_solution(
        planner_factory: Dict, 
        result: Any, 
        limits: List[List[float]] = [[-6, 6], [-6, 6]], 
        interval: int = 50, 
        step_size: float = 0.25, 
        node_size: int = 20, 
        progress_widget: Optional[IntProgress] = None
    ) -> Optional[FuncAnimation]:
        """
        Generates an HTML5 Animation with Split-Screen view (Task Space & Graph Search).

        Args:
            planner_factory (dict): Dictionary containing planner factory methods.
            result (ResultCollection): Object containing planner, solution, and benchmark data.
            limits (list): List of [min, max] for X and Y axes.
            interval (int): Delay between frames in milliseconds.
            step_size (float): Interpolation resolution (lower = smoother but slower).
            node_size (int): Visual size of nodes in the graph plot.
            progress_widget (IntProgress, optional): Widget to display rendering progress in UI.

        Returns:
            FuncAnimation: The matplotlib animation object, or None if no path exists.
        """        
        plt.rcParams['animation.embed_limit'] = 200

        # --- A. Data Extraction & Validation ---
        planner = result.planner
        solution = result.solution
        actions = result.actions
        graph = getattr(planner, 'graph', None)

        if solution is None or len(solution) == 0:
            print(f"[IPAnimator] No path found for {result.plannerFactoryName}. Skipping animation.")
            return None
        
        if graph is None:
             # Fallback if the planner doesn't expose a graph (e.g. some RRT variants might differ)
             # Assuming graph copy is necessary for visualization
             graph = planner.graph.copy()
        else:
             graph = graph.copy()

        cc = result.benchmark.collisionChecker
        
        # Resolve Node IDs to Configuration Coordinates if necessary
        config_path = []
        if len(solution) > 0 and isinstance(solution[0], (int, np.integer, str)):
            for node_id in solution:
                if node_id in graph.nodes:
                    config_path.append(graph.nodes[node_id]['pos'])
        else:
            config_path = solution

        if not config_path:
            return None

        # --- B. Trajectory Generation ---
        full_trajectory = IPAnimator._get_trajectory_with_state(
            config_path, actions, cc, step_size=step_size
        )
        
        print(f"[IPAnimator] Generating Animation for '{result.plannerFactoryName} - {result.benchmark.name}' "
              f"({len(full_trajectory)} frames)...")

        # Initialize UI Progress Bar
        if progress_widget:
            progress_widget.value = 0
            progress_widget.max = len(full_trajectory)
            progress_widget.description = f"0/{len(full_trajectory)}"
            progress_widget.bar_style = 'info'

        # --- C. Figure Setup ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        plt.close(fig)  # Prevents double display in notebooks

        # Set Titles
        ax1.set_title(f"Task Space: {result.benchmark.name}")
        ax2.set_title("Graph Projection (X-Y)")

        # Set Axis Limits
        for ax in [ax1, ax2]:
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

        # --- D. Background Plotting (Graph) ---
        # Draw the underlying graph structure on the right axis (ax2)
        plot_func = planner_factory[result.plannerFactoryName][2]
        plot_func(
            result.planner, 
            result.solution, 
            actions=actions, 
            ax=ax2, 
            nodeSize=node_size, 
            plot_only_solution=False, 
            plot_robot=False
        )

        # Draw Action Markers (PICK/PLACE) explicitly on ax2
        if actions:
            for step_idx, (act_type, _) in actions.items():
                if step_idx < len(solution):
                    node_id = solution[step_idx]
                    if node_id in graph.nodes:
                        pos = graph.nodes[node_id]['pos']
                        if act_type == "PICK":
                            ax2.plot(pos[0], pos[1], 's', color='lime', markersize=10, zorder=10)
                            ax2.text(pos[0], pos[1]+0.5, 'PICK', color='green', fontsize=8, 
                                     ha='center', bbox=dict(facecolor='white', alpha=0.6, pad=0.5))
                        elif act_type == "PLACE":
                            ax2.plot(pos[0], pos[1], 'X', color='red', markersize=10, zorder=10)
                            ax2.text(pos[0], pos[1]+0.5, 'PLACE', color='red', fontsize=8, 
                                     ha='center', bbox=dict(facecolor='white', alpha=0.6, pad=0.5))

        # Marker for current robot state on the graph
        current_pos_marker, = ax2.plot([], [], 'ro', markersize=8, zorder=10, label='Current')
        ax2.legend(loc='upper right')

        # --- E. Animation Update Loop ---
        def update(frame):
            # Update Progress Bar
            if progress_widget:
                progress_widget.value = frame + 1
                progress_widget.description = f"{frame + 1}/{len(full_trajectory)}"

            # 1. Clear and Reset Left Axis (Task Space)
            ax1.clear()
            ax1.set_xlim(limits[0])
            ax1.set_ylim(limits[1])
            ax1.set_aspect('equal')
            ax1.grid(True, alpha=0.3)
            ax1.set_title(f"Step {frame}/{len(full_trajectory)}")

            # 2. Extract Frame Data
            config, held_obj, world_obj_poly = full_trajectory[frame]

            # 3. Update Collision Checker State
            if held_obj is not None:
                cc.attach_object(held_obj)
            else:
                cc.detach_object()

            # 4. Draw Scene
            cc.drawObstacles(ax1)
            
            # Draw Robot (orange)
            cc.drawRobot(config, ax1, alpha=0.9, color='orange')

            # Draw "Loose" Object in World (green)
            if world_obj_poly is not None:
                ox, oy = world_obj_poly.exterior.xy
                ax1.fill(ox, oy, fc='lime', alpha=0.9, ec='black', 
                         linewidth=1, linestyle='-', label="Loose Object")

            # 5. Update Graph Marker (Right Axis)
            current_pos_marker.set_data([config[0]], [config[1]])
            
            return []

        # Create Animation Object
        ani = FuncAnimation(
            fig, update, frames=len(full_trajectory), 
            interval=interval, repeat_delay=1000
        )
        return ani
    
    @staticmethod
    def create_interactive_viewer(
        planner_factory: Dict, 
        result_list: List[Any], 
        limits: List[List[float]] = [[-6, 6], [-6, 6]]
    ) -> VBox:
        """
        Creates an interactive Jupyter Widget to view and save animations.

        Args:
            planner_factory (dict): Dictionary of planner factories.
            result_list (list): List of ResultCollection objects.
            limits (list): Axis limits for visualization.

        Returns:
            VBox: A widget container holding the dropdowns and visual output.
        """
        # 1. Filter valid results
        successful_results = [res for res in result_list if res.solution]
        
        if not successful_results:
            print("No paths found to animate.")
            return VBox([Label("No successful paths found.")])

        # Prepare Dropdown Options
        options = [("--- Nothing Selected ---", -1)] + \
                  [(f"{res.plannerFactoryName} - {res.benchmark.name}", i) 
                   for i, res in enumerate(successful_results)]

        # 2. Initialize Widgets
        dropdown = Dropdown(
            options=options, 
            value=-1, 
            description='Result:', 
            layout=Layout(width='50%')
        )
        btn_save_mp4 = Button(
            description='Save .mp4', icon='save', 
            disabled=True, button_style='info'
        )
        btn_save_gif = Button(
            description='Save .gif', icon='save', 
            disabled=True, button_style='warning'
        )

        progress_bar = IntProgress(
            value=0, min=0, max=100,
            description='Idle',
            bar_style='', 
            orientation='horizontal',
            layout=Layout(width='100%')
        )
        
        anim_output = Output()
        msg_output = Output()

        # State container to hold the current animation object
        state = {'current_anim': None}

        # 3. Event Callbacks
        def on_selection_change(change):
            idx = change['new']
            
            anim_output.clear_output()
            msg_output.clear_output()
            state['current_anim'] = None
            
            if idx == -1:
                btn_save_mp4.disabled = True
                btn_save_gif.disabled = True
                with anim_output:
                    print("Please select a result.")
                return

            btn_save_mp4.disabled = True
            btn_save_gif.disabled = True
            
            res = successful_results[idx]
            
            with anim_output:
                print("Calculating animation frames...")
                try:
                    ani = IPAnimator.animate_solution(
                        planner_factory, res, limits=limits, 
                        interval=75, step_size=0.5, 
                        progress_widget=progress_bar
                    )
                    state['current_anim'] = ani

                    clear_output(wait=True)
                    if ani:
                        print("Rendering HTML JS (this may take a moment)...")
                        # Rendering the JSHTML triggers the animation progress
                        display(HTML(ani.to_jshtml()))
                        
                        progress_bar.bar_style = 'success'
                        progress_bar.description = 'Done'
                        
                        btn_save_mp4.disabled = False
                        btn_save_gif.disabled = False
                except Exception as e:
                    print(f"Error during animation generation: {e}")
                    progress_bar.bar_style = 'danger'

        def save_file(b, fmt):
            if state['current_anim'] is None:
                return
            
            idx = dropdown.value
            res = successful_results[idx]
            timestamp = datetime.datetime.now().strftime('%Y%m%d')
            sanitized_name = f"{res.plannerFactoryName}_{res.benchmark.name}".replace(" ", "_")
            filename = f"{timestamp}_{sanitized_name}.{fmt}"
            
            with msg_output:
                print(f"Saving {filename}...")
            
            try:
                if fmt == 'mp4':
                    # Requires ffmpeg installed on the system
                    state['current_anim'].save(filename, writer='ffmpeg', dpi=100)
                else:
                    # Requires Pillow/ImageMagick
                    state['current_anim'].save(filename, writer='pillow', fps=15)
                
                with msg_output:
                    print(f"Successfully saved: {filename}")
            except Exception as e:
                with msg_output:
                    print(f"Save Error: {e}")

        # 4. Link Events
        dropdown.observe(on_selection_change, names='value')
        btn_save_mp4.on_click(lambda b: save_file(b, 'mp4'))
        btn_save_gif.on_click(lambda b: save_file(b, 'gif'))

        # 5. Assemble Layout
        return VBox([
            HBox([dropdown, btn_save_mp4, btn_save_gif]),
            progress_bar,
            msg_output,
            anim_output
        ])