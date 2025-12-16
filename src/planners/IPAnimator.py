# src/planners/IPAnimator.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import networkx as nx

class IPAnimator:
    """
    Helper class to generate smooth animations for Robot Path Planning results.
    """

    @staticmethod
    def _interpolate_line(startPos, endPos, step_size=0.2):
        """Interpolates linearly between two configurations."""
        start = np.array(startPos)
        end = np.array(endPos)
        dist = np.linalg.norm(end - start)
        
        if dist < 1e-6:
            return [start]
        
        n_steps = int(dist / step_size)
        if n_steps < 1: n_steps = 1
        
        steps = []
        for i in range(n_steps):
            alpha = i / n_steps
            interp = start * (1 - alpha) + end * alpha
            steps.append(interp)
        
        return steps

    @staticmethod
    def _get_interpolated_path(config_path, step_size=0.2):
        """Creates a smooth path from a list of waypoints."""
        smooth_path = []
        for i in range(len(config_path) - 1):
            segment = IPAnimator._interpolate_line(config_path[i], config_path[i+1], step_size)
            smooth_path.extend(segment)
        
        smooth_path.append(np.array(config_path[-1]))
        return smooth_path

    @staticmethod
    def animate_solution(plannerFactory, result, limits=(-10, 25), interval=50, step_size=0.25, nodeSize=20):
        """
        Generates the HTML5 Animation with Split-Screen (Task Space & Graph).
        
        Args:
            result_obj: ResultCollection object containing planner, solution, benchmark.
            limits: Tuple (min, max) for the plot axis limits.
            interval: Animation speed in ms.
            step_size: Interpolation step size (lower = smoother but more frames).
        
        Returns:
            HTML object containing the JS animation.
        """        

        plt.rcParams['animation.embed_limit'] = 100

        # A. Path Validation & Extraction
        planner = result.planner
        solution = result.solution
        graph = planner.graph.copy()
        if solution is None or len(solution) == 0:
            print(f"[IPAnimator] No path for {result.plannerFactoryName}")
            return None
        
        cc = result.benchmark.collisionChecker
        
        # Convert Node IDs to Configs if necessary
        config_path = []
        if isinstance(solution[0], (int, np.integer, str)):
            # Assuming Node IDs
            for node_id in solution:
                try:
                    config_path.append(graph.nodes[node_id]['pos'])
                except KeyError:
                    # Fallback if pos is missing
                    pass
        else:
            config_path = solution

        if not config_path:
            return None

        # B. Interpolation
        full_trajectory = IPAnimator._get_interpolated_path(config_path, step_size=step_size)
        
        print(f"Generating Animation for '{result.plannerFactoryName} - {result.benchmark.name}' ({len(full_trajectory)} frames)...")

        # C. Setup Figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        plt.close(fig) # Prevent static output

        # Titles
        ax1.set_title(f"Task Space: {result.benchmark.name}")
        ax2.set_title(f"Graph Projection (X-Y)")

        # Limits
        for ax in [ax1, ax2]:
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

        # --- Plot 2: Graph Projection (Background) ---
        plannerFactory[result.plannerFactoryName][2](result.planner, result.solution, ax=ax, nodeSize=20, plot_only_solution=False, plot_robot=False)

        # Marker for current state
        current_pos_marker, = ax2.plot([], [], 'ro', markersize=8, zorder=10, label='Current')
        ax2.legend(loc='upper right')

        # --- D. Animation Loop ---
        def update(frame):
            # 1. Update Workspace (Left)
            ax1.clear()
            # Reset properties after clear
            ax1.set_xlim(limits)
            ax1.set_ylim(limits)
            ax1.set_aspect('equal')
            ax1.grid(True, alpha=0.3)
            ax1.set_title(f"Step {frame}/{len(full_trajectory)}")

            # Draw
            cc.drawObstacles(ax1)
            
            config = full_trajectory[frame]
            cc.drawRobot(config, ax1, alpha=0.9, color='orange')

            # 2. Update Graph Marker (Right)
            current_pos_marker.set_data([config[0]], [config[1]])
            
            return []

        # E. Render
        ani = FuncAnimation(fig, update, frames=len(full_trajectory), interval=interval, repeat_delay=1.0)
        return HTML(ani.to_jshtml())