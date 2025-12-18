# src/planners/IPAnimator.py

import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display, clear_output
import networkx as nx
from shapely.geometry import Polygon
from ipywidgets import Button, Dropdown, VBox, HBox, Output, Layout, Label, IntProgress

class IPAnimator:
    """
    Helper class to generate smooth animations for Robot Path Planning results.
    Updated for Pick & Place Visualization.
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

    # @staticmethod
    # def _get_interpolated_path(config_path, step_size=0.2):
    #     """Creates a smooth path from a list of waypoints."""
    #     smooth_path = []
    #     for i in range(len(config_path) - 1):
    #         segment = IPAnimator._interpolate_line(config_path[i], config_path[i+1], step_size)
    #         smooth_path.extend(segment)
        
    #     smooth_path.append(np.array(config_path[-1]))
    #     return smooth_path

    @staticmethod
    def _get_trajectory_with_state(config_path, actions, cc, step_size=0.2):
        """
        Creates a smooth path containing configuration AND gripper state for every frame.
        
        Args:
            config_path: List of configurations (nodes).
            actions: Dictionary { index: ("ACTION", shape) }
            
        Returns:
            trajectory: List of tuples [(config, held_object_shape), ...]
        """
        trajectory = []
        
        # --- 1. Initiale Objekt-Position bestimmen ---
        # Wir schauen, wo der ERSTE Pick stattfindet. Dort liegt das Objekt am Anfang.
        current_world_obj_poly = None
        current_held_object = None # Am Start trägt der Roboter nichts

        first_pick_index = -1
        object_shape_def = None

        if actions:
            # Suche nach dem ersten "PICK"
            for idx, (act, shape) in actions.items():
                if act == "PICK":
                    first_pick_index = idx
                    object_shape_def = shape
                    break

        if first_pick_index != -1 and first_pick_index < len(config_path):
            # Wir simulieren kurz, dass das Objekt dort gegriffen ist, um die Position zu bekommen
            pick_config = config_path[first_pick_index]
            
            # Helper: Objekt kurz anhängen, Geometrie berechnen, Polygon kopieren, abhängen
            cc.attach_object(object_shape_def)
            geo = cc.get_robot_geometry(pick_config)
            if geo['held_object'] is not None:
                current_world_obj_poly = geo['held_object'] # Das ist jetzt das Polygon in Weltkoordinaten
            cc.detach_object()
        
        # --- 2. Pfad interpolieren ---
        for i in range(len(config_path) - 1):
            start_conf = config_path[i]
            end_conf = config_path[i+1]
            
            # Interpolate Segment
            segment_configs = IPAnimator._interpolate_line(start_conf, end_conf, step_size)
            
            # Append frames with CURRENT state
            for conf in segment_configs:
                # Frame speichern: (RoboterConfig, WasHängtAmArm, WasLiegtInDerWelt)
                trajectory.append((conf, current_held_object, current_world_obj_poly))
            
            # --- STATE TRANSITION CHECK ---
            # Check if an action happens at the target node (i+1) of this movement
            target_node_index = i + 1
            if actions and target_node_index in actions:
                action_type, obj_shape = actions[target_node_index]
                target_config = config_path[target_node_index]
                
                if action_type == "PICK":
                    # Roboter nimmt Objekt -> Welt leer, Arm voll
                    current_held_object = obj_shape
                    current_world_obj_poly = None

                elif action_type == "PLACE":
                    # Roboter legt ab -> Arm leer, Welt voll (an der aktuellen Position)
                    
                    # Berechne, wo das Objekt jetzt liegt
                    # Wir müssen es kurz attachen, um die Kinematik zu nutzen
                    cc.attach_object(current_held_object) # Das Objekt, das wir gerade noch hatten
                    geo = cc.get_robot_geometry(target_config)
                    current_world_obj_poly = geo['held_object']
                    cc.detach_object() # Cleanup für den CC, state wird über Variable gesteuert
                    
                    current_held_object = None

        # Letzten Punkt anfügen
        trajectory.append((config_path[-1], current_held_object, current_world_obj_poly))
        
        return trajectory

    @staticmethod
    def animate_solution(plannerFactory, result, limits=(-10, 25), interval=50, step_size=0.25, nodeSize=20, progress_widget=None):
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

        plt.rcParams['animation.embed_limit'] = 200

        # A. Path Validation & Extraction
        planner = result.planner
        solution = result.solution
        actions = result.actions
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
        # This creates a list of (config, shape) tuples
        full_trajectory = IPAnimator._get_trajectory_with_state(config_path, actions, cc, step_size=step_size)
        
        print(f"[IPAnimator] Generating Animation for '{result.plannerFactoryName} - {result.benchmark.name}' ({len(full_trajectory)} frames)...")

        # --- NEU: Progress Bar initialisieren ---
        if progress_widget:
            progress_widget.value = 0
            progress_widget.max = len(full_trajectory)
            progress_widget.description = f"0/{len(full_trajectory)}"
            progress_widget.bar_style = 'info'
        # ----------------------------------------

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
        plannerFactory[result.plannerFactoryName][2](result.planner, result.solution, actions=actions, ax=ax, nodeSize=20, plot_only_solution=False, plot_robot=False)

        # Draw Action Markers manually on ax2 (Right side)
        if actions:
            for step_idx, (act_type, _) in actions.items():
                if step_idx < len(solution):
                    node_id = solution[step_idx]
                    if node_id in graph.nodes:
                        pos = graph.nodes[node_id]['pos']
                        if act_type == "PICK":
                            ax2.plot(pos[0], pos[1], 's', color='lime', markersize=10, zorder=10)
                            ax2.text(pos[0], pos[1]+0.5, 'PICK', color='green', fontsize=8, ha='center', bbox=dict(facecolor='white', alpha=0.6, pad=0.5))
                        elif act_type == "PLACE":
                            ax2.plot(pos[0], pos[1], 'X', color='red', markersize=10, zorder=10)
                            ax2.text(pos[0], pos[1]+0.5, 'PLACE', color='red', fontsize=8, ha='center', bbox=dict(facecolor='white', alpha=0.6, pad=0.5))

        # Marker for current state
        current_pos_marker, = ax2.plot([], [], 'ro', markersize=8, zorder=10, label='Current')
        ax2.legend(loc='upper right')

        # --- D. Animation Loop ---
        def update(frame):
            # --- NEU: Progress Update ---
            if progress_widget:
                progress_widget.value = frame + 1
                progress_widget.description = f"{frame + 1}/{len(full_trajectory)}"
            # -----------------------------

            # 1. Update Workspace (Left)
            ax1.clear()
            # Reset properties after clear
            ax1.set_xlim(limits)
            ax1.set_ylim(limits)
            ax1.set_aspect('equal')
            ax1.grid(True, alpha=0.3)
            ax1.set_title(f"Step {frame}/{len(full_trajectory)}")

            # Unpack: Config, HeldObject, WorldObject
            config, held_obj, world_obj_poly = full_trajectory[frame]

            # 1. CollisionChecker State setzen (für den Roboter)
            if held_obj is not None:
                cc.attach_object(held_obj)
            else:
                cc.detach_object()

            # 2. Zeichnen
            cc.drawObstacles(ax1)

            # A) Roboter zeichnen (inkl. Held Object falls vorhanden)
            cc.drawRobot(config, ax1, alpha=0.9, color='orange')

            # B) Freies Objekt zeichnen (Falls es in der Welt liegt)
            if world_obj_poly is not None:
                ox, oy = world_obj_poly.exterior.xy
                # Wir zeichnen das "loose" Objekt etwas dunkler/transparenter grün
                ax1.fill(ox, oy, fc='lime', alpha=0.9, ec='black', linewidth=1, linestyle='-', label="Loose Object")

            # 3. Update Graph Marker (Right)
            current_pos_marker.set_data([config[0]], [config[1]])
            
            return []

        # E. Render
        ani = FuncAnimation(fig, update, frames=len(full_trajectory), interval=interval, repeat_delay=1.0)
        return ani
    
    
    # --- NEUE UI METHODE ---
    @staticmethod
    def create_interactive_viewer(plannerFactory, resultList, limits=(-10, 25)):
        """
        Creates and returns an interactive widget (VBox) to select, view, and save animations.
        """
        # 1. Daten vorbereiten
        successful_results = [res for res in resultList if res.solution != []]
        
        if not successful_results:
            print("No paths found to animate.")
            return VBox([Label("No successful paths found.")])

        options = [("--- Nothing Selected ---", -1)] + \
                  [(f"{res.plannerFactoryName} - {res.benchmark.name}", i) for i, res in enumerate(successful_results)]

        # 2. Widgets initialisieren
        dropdown = Dropdown(options=options, value=-1, description='Result:', layout=Layout(width='50%'))
        btn_save_mp4 = Button(description='Save .mp4', icon='save', disabled=True, button_style='info')
        btn_save_gif = Button(description='Save .gif', icon='save', disabled=True, button_style='warning')

        # --- NEU: Progress Bar ---
        progress_bar = IntProgress(
            value=0,
            min=0,
            max=100,
            description='Idle',
            bar_style='', # 'success', 'info', 'warning', 'danger' or ''
            orientation='horizontal',
            layout=Layout(width='100%')
        )
        # -------------------------
        
        anim_output = Output()
        msg_output = Output()

        # Mutable Container für das aktuelle Animationsobjekt (in Closure verfügbar machen)
        state = {'current_anim': None}

        # 3. Callbacks definieren
        def on_selection_change(change):
            idx = change['new']
            
            anim_output.clear_output()
            msg_output.clear_output()
            state['current_anim'] = None
            
            if idx == -1:
                btn_save_mp4.disabled = True
                btn_save_gif.disabled = True
                with anim_output: print("Please select a result.")
                return

            btn_save_mp4.disabled = True
            btn_save_gif.disabled = True
            
            res = successful_results[idx]
            
            with anim_output:
                print("Calculating animation...")
                # Animation erstellen
                try:
                    ani = IPAnimator.animate_solution(plannerFactory, res, limits=limits, interval=75, step_size=0.5, progress_widget=progress_bar)
                    state['current_anim'] = ani

                    clear_output(wait=True)
                    if ani:
                        print("Rendering HTML (this takes time)...")
                        # Der Progress Bar füllt sich während .to_jshtml() aufgerufen wird
                        display(HTML(ani.to_jshtml()))
                        
                        # Wenn fertig:
                        progress_bar.bar_style = 'success'
                        progress_bar.description = 'Done'
                        
                        btn_save_mp4.disabled = False
                        btn_save_gif.disabled = False
                except Exception as e:
                    print(f"Error: {e}")
                    progress_bar.bar_style = 'danger'

        def save_file(b, fmt):
            if state['current_anim'] is None: return
            idx = dropdown.value
            res = successful_results[idx]
            filename = f"{datetime.datetime.now().strftime('%Y%m%d')}_{res.plannerFactoryName}_{res.benchmark.name}".replace(" ", "_") + f".{fmt}"
            
            with msg_output:
                print(f"Saving {filename}...")
            
            try:
                if fmt == 'mp4':
                    state['current_anim'].save(filename, writer='ffmpeg', dpi=100)
                else:
                    state['current_anim'].save(filename, writer='pillow', fps=15)
                with msg_output: print(f"Saved: {filename}")
            except Exception as e:
                with msg_output: print(f"Save Error: {e}")

        # 4. Verknüpfung
        dropdown.observe(on_selection_change, names='value')
        btn_save_mp4.on_click(lambda b: save_file(b, 'mp4'))
        btn_save_gif.on_click(lambda b: save_file(b, 'gif'))

        # 5. UI zurückgeben (mit Progress Bar)
        return VBox([
            HBox([dropdown, btn_save_mp4, btn_save_gif]),
            progress_bar, # <--- Hinzugefügt
            msg_output,
            anim_output
        ])