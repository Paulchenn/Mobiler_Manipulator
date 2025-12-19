import networkx as nx
import numpy as np

class MultiGoalPlannerRunner:
    """
    Führt einen Pfadplaner sequenziell für mehrere Ziele aus und berücksichtigt
    dabei Pick & Place Aktionen (definiert als Tupel in der goalList).
    """

    # Konfiguration für den Vor-Krampf (in Metern)
    DEFAULT_APPROACH_VEC = [0.6, 0.0]
    # Wie viele Zwischenschritte für die lineare Bewegung generiert werden sollen
    LINEAR_STEPS = 5
    
    @staticmethod
    def get_standoff_config(target_config, offset_vector):
        """
        Berechnet den Standoff-Punkt basierend auf einem lokalen Offset-Vektor.
        offset_vector = [dx, dy] (relativ zur Ausrichtung theta)
        """
        x_target, y_target, theta = target_config[0], target_config[1], target_config[2]

        dx_local, dy_local = offset_vector

        # Rotation des lokalen Offset-Vektors in das Welt-System
        # Wir subtrahieren den Vektor, da wir den Punkt *vor* dem Ziel wollen
        # Formel: P_standoff = P_target - R(theta) * V_offset

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Rotierter Vektor
        dx_world = dx_local * cos_t - dy_local * sin_t
        dy_world = dx_local * sin_t + dy_local * cos_t

        # Standoff berechnen (Target MINUS Offset)
        standoff_x = x_target - dx_world
        standoff_y = y_target - dy_world
        
        # Konfiguration kopieren und Koordinaten überschreiben
        standoff_config = target_config.copy()
        standoff_config[0] = standoff_x
        standoff_config[1] = standoff_y
        
        return standoff_config
    
    @staticmethod
    def interpolate_linear(start, end, steps):
        """Erzeugt eine lineare Interpolation zwischen zwei Konfigurationen."""
        trajectory = []
        s = np.array(start)
        e = np.array(end)
        for i in range(1, steps + 1): # Start ist exklusive (haben wir schon), Ende inclusive
            alpha = i / steps
            point = s * (1 - alpha) + e * alpha
            trajectory.append(point)
        return trajectory

    @staticmethod
    def run_benchmark(planner, benchmark, config):
        """
        Führt die Planung durch.
        
        Erwartet im 'benchmark'-Objekt:
            - benchmark.goalList: Liste von Zielen. 
              Format A (Neu): [ (PosA, "PICK"), (PosB, "PLACE"), (PosC, "MOVE") ]
              Format B (Alt): [ PosA, PosB, PosC ] (Aktionen sind dann implizit None)
            - benchmark.objectShape (list): Polygon-Punkte des Objekts [[x,y], ...]
        """
        
        # --- 1. SETUP & CONTAINER ---
        full_relabeled_path = []
        full_roadmap_graph = nx.Graph()
        full_colliding_edges = []       
        full_non_colliding_edges = []   

        # NEU: Dictionary für Events {Index im Pfad : "ACTION"}
        action_events = {}
        
        current_start = benchmark.startList 
        targets = benchmark.goalList

        # object_shape = getattr(benchmark, 'objectShape', None)
        # print(f"        [Setup] Object shape (benchmark): {object_shape}")
        object_shape = planner._collisionChecker.get_object_shape()
        # print(f"        [Setup] Object shape (planner): {object_shape}")
        
        last_segment_end_node = None
        current_world_obstacle_poly = None

        # Status Tracking Variablen
        failed_segment_index = -1 # -1 bedeutet Success
        failed_reason = ""
        
        # Sicherstellen, dass der Greifer am Anfang leer ist
        if hasattr(planner._collisionChecker, 'detach_object'):
            planner._collisionChecker.detach_object()

        # --- 2. INITIALES OBJEKT PLATZIEREN (NEU!) ---
        # Wir suchen den ersten "PICK" Befehl und platzieren das Objekt dort als Hindernis.
        if object_shape is not None:
            for t in targets:
                # Parsing Logic (Quick check)
                # Check auf Tupel-Länge 2 oder 3
                if isinstance(t, (tuple, list)) and len(t) >= 2 and t[1] == "PICK":
                    pick_config = t[0]
                    
                    # Trick: Objekt kurz anhängen, um Welt-Position zu berechnen
                    planner._collisionChecker.attach_object(object_shape)
                    geo = planner._collisionChecker.get_robot_geometry(pick_config)
                    poly = geo.get('held_object')
                    planner._collisionChecker.detach_object()
                    
                    if poly is not None:
                        # Als Hindernis hinzufügen
                        print(f"        [Setup] Initial object placed at Pick-Location as obstacle.")
                        planner._collisionChecker.obstacles.append(poly)
                        current_world_obstacle_poly = poly
                    break # Wir nehmen an, das Objekt liegt beim ersten Pick

        # --- 3. PLANUNGSSCHLEIFE ---
        for i, target_entry in enumerate(targets):
            
            # 1. Ziel analysieren
            actual_target_coords = None
            current_action = "MOVE"
            current_offset = MultiGoalPlannerRunner.DEFAULT_APPROACH_VEC

            # print(target_entry)
            if isinstance(target_entry, (tuple, list)):
                actual_target_coords = target_entry[0]

                # Fall 2: (Config, Action) -> Länge 2
                if len(target_entry) >= 2: current_action = target_entry[1]
                # Fall 1: (Config, Action, Offset) -> Länge 3
                if len(target_entry) >= 3: current_offset = target_entry[2]

            # Fall 3: Nur Config -> Länge wurscht, interpretieren als Koordinaten
            else:
                actual_target_coords = target_entry

            # 2. Planungs-Ziel bestimmen (Standoff oder direkt?)
            # Wenn PICK oder PLACE -> Fahre erst zum Standoff-Punkt
            if current_action in ["PICK", "PLACE"]:
                planner_goal_coords = MultiGoalPlannerRunner.get_standoff_config(actual_target_coords, current_offset)
                print(f"        [Runner] Segment {i}: {current_action} with offset {current_offset}. Planning to Standoff.")
            else:
                planner_goal_coords = actual_target_coords
                print(f"        [Runner] Segment {i}: MOVE. Planning directly to target.")

            current_goal_list_for_planner = [planner_goal_coords]

            # 3. GLOBAL PLANNING (Start -> Standoff)
            # print(f"actual_target_coords: {actual_target_coords}")
            # print(f"current_start: {current_start}")
            # print(f"current_goal_list_for_planner: {current_goal_list_for_planner}")
            # print(f"config: {config}")

            # if i == 2:
            #     print("My stop")
            #     break
            segment = planner.planPath(current_start, current_goal_list_for_planner, config)

            # --- FEHLERBEHANDLUNG (NEU: BREAK STATT EXCEPTION) ---
            if segment is None or len(segment) == 0:
                print(f"        [WARNING] No path found in segment {i} ({current_action})! Stopping execution here.")
                failed_segment_index = i
                failed_reason = f"Failed at Segment {i}: {current_action}"
                break # <--- HIER BRECHEN WIR AB
            
            # --- Graph Merging (Standard Path) ---
            current_graph = planner.graph
            mapping = {node: f"s{i}_{node}" for node in current_graph.nodes()}
            relabeled_segment = [mapping[n] for n in segment]
            
            relabeled_graph = nx.relabel_nodes(current_graph, mapping)
            full_roadmap_graph = nx.compose(full_roadmap_graph, relabeled_graph)
            
            if hasattr(planner, 'collidingEdges'):
                for u, v in planner.collidingEdges:
                    if u in mapping and v in mapping:
                        full_colliding_edges.append((mapping[u], mapping[v]))
            if hasattr(planner, 'nonCollidingEdges'):
                for u, v in planner.nonCollidingEdges:
                    if u in mapping and v in mapping:
                        full_non_colliding_edges.append((mapping[u], mapping[v]))

            if last_segment_end_node is not None:
                full_roadmap_graph.add_edge(last_segment_end_node, relabeled_segment[0], connection="virtual") 

            # Letzter Knoten des geplanten Pfades (das ist der Standoff Punkt!)
            full_relabeled_path.extend(relabeled_segment)
            last_segment_end_node = relabeled_segment[-1]

            # --- ACTION HANDLING ---
            # Wo im Gesamtpfad sind wir JETZT (am Ende dieses Segments)?
            # current_end_index = len(full_relabeled_path) - 1

            # 4. LINEAR APPROACH & RETREAT LOGIC (Nur bei Pick/Place)
            if current_action in ["PICK", "PLACE"]:
                
                # A) Hinfahren (Standoff -> Target)
                # --------------------------------
                approach_path = MultiGoalPlannerRunner.interpolate_linear(planner_goal_coords, actual_target_coords, MultiGoalPlannerRunner.LINEAR_STEPS)
                
                prev_node_id = last_segment_end_node
                
                for k, pt in enumerate(approach_path):
                    # Fake Node erstellen
                    node_id = f"s{i}_approach_in_{k}"
                    full_roadmap_graph.add_node(node_id, pos=pt)
                    full_roadmap_graph.add_edge(prev_node_id, node_id, connection="linear")
                    full_relabeled_path.append(node_id)
                    prev_node_id = node_id
                
                # Jetzt sind wir AM Ziel (actual_target_coords). Hier passiert die Action.
                current_end_index = len(full_relabeled_path) - 1
                poly_to_activate_after_retreat = None
                
                # --- ACTION EXECUTION ---
                if current_action == "PICK":
                    action_events[current_end_index] = ("PICK", object_shape)
                    
                    if current_world_obstacle_poly in planner._collisionChecker.obstacles:
                        planner._collisionChecker.obstacles.remove(current_world_obstacle_poly)
                        print(f"        [Action] PICK: Removed static object.")
                        current_world_obstacle_poly = None

                    if object_shape is not None:
                        print(f"        [Action] PICK: Grasping.")
                        planner._collisionChecker.attach_object(object_shape)
                
                elif current_action == "PLACE":
                    action_events[current_end_index] = ("PLACE", None)
                    
                    # Position für Hindernis berechnen (am Zielpunkt)
                    geo = planner._collisionChecker.get_robot_geometry(actual_target_coords)
                    placed_poly = geo.get('held_object')
                    
                    print(f"        [Action] PLACE: Releasing.")
                    planner._collisionChecker.detach_object()

                    if placed_poly is not None:
                        print(f"        [Action] PLACE: Object becomes obstacle.")
                        # Ghost Mode: Erst nach Retreat aktivieren
                        poly_to_activate_after_retreat = placed_poly

                # B) Zurückfahren (Target -> Standoff)
                # ------------------------------------
                # Wir fahren zurück zum planner_goal_coords (Standoff)
                retreat_path = MultiGoalPlannerRunner.interpolate_linear(actual_target_coords, planner_goal_coords, MultiGoalPlannerRunner.LINEAR_STEPS)
                
                for k, pt in enumerate(retreat_path):
                    node_id = f"s{i}_approach_out_{k}"
                    full_roadmap_graph.add_node(node_id, pos=pt)
                    full_roadmap_graph.add_edge(prev_node_id, node_id, connection="linear")
                    full_relabeled_path.append(node_id)
                    prev_node_id = node_id

                # Jetzt Objekt aktivieren
                if poly_to_activate_after_retreat is not None:
                    planner._collisionChecker.obstacles.append(poly_to_activate_after_retreat)
                    current_world_obstacle_poly = poly_to_activate_after_retreat
                
                # Der Roboter steht jetzt wieder am Standoff Punkt.
                # Das ist der Start für die nächste Runde.
                current_start = [planner_goal_coords]
                last_segment_end_node = prev_node_id

            else:
                # MOVE Case: Wir sind am Ziel angekommen und bleiben da.
                current_start = [actual_target_coords]

        # --- CLEANUP & FINALIZE ---
        if full_relabeled_path:
            real_start = full_relabeled_path[0]
            if real_start in full_roadmap_graph.nodes:
                full_roadmap_graph.add_node("start", pos=full_roadmap_graph.nodes[real_start]['pos'])
            real_goal = full_relabeled_path[-1]
            if real_goal in full_roadmap_graph.nodes:
                full_roadmap_graph.add_node("goal", pos=full_roadmap_graph.nodes[real_goal]['pos'])

        planner.graph = full_roadmap_graph
        planner.collidingEdges = full_colliding_edges         
        planner.nonCollidingEdges = full_non_colliding_edges  
        
        if hasattr(planner._collisionChecker, 'detach_object'):
            planner._collisionChecker.detach_object()
        if current_world_obstacle_poly in planner._collisionChecker.obstacles:
            planner._collisionChecker.obstacles.remove(current_world_obstacle_poly)

        # NEUER RETURN WERT: Status Dictionary
        status = {
            "success": (failed_segment_index == -1),
            "fail_segment": failed_segment_index,
            "fail_reason": failed_reason,
            "total_segments": len(targets)
        }
        
        return full_relabeled_path, action_events, status