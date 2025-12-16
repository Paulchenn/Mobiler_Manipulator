import networkx as nx

class MultiGoalPlannerRunner:
    """
    Führt einen Pfadplaner sequenziell für mehrere Ziele aus und berücksichtigt
    dabei Pick & Place Aktionen (definiert als Tupel in der goalList).
    """

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
        object_shape = planner._collisionChecker.get_object_shape()
        
        last_segment_end_node = None
        
        # Sicherstellen, dass der Greifer am Anfang leer ist
        if hasattr(planner._collisionChecker, 'detach_object'):
            planner._collisionChecker.detach_object()

        # --- 2. INITIALES OBJEKT PLATZIEREN (NEU!) ---
        # Wir suchen den ersten "PICK" Befehl und platzieren das Objekt dort als Hindernis.
        if object_shape is not None:
            for t in targets:
                # Parsing Logic (Quick check)
                if isinstance(t, (tuple, list)) and len(t) == 2 and t[1] == "PICK":
                    pick_config = t[0]
                    
                    # Trick: Objekt kurz anhängen, um Welt-Position zu berechnen
                    planner._collisionChecker.attach_object(object_shape)
                    geo = planner._collisionChecker.get_robot_geometry(pick_config)
                    poly = geo.get('held_object')
                    planner._collisionChecker.detach_object()
                    
                    if poly is not None:
                        # Als Hindernis hinzufügen
                        print(f"      [Setup] Initial object placed at Pick-Location as obstacle.")
                        planner._collisionChecker.obstacles.append(poly)
                        current_world_obstacle_poly = poly
                    break # Wir nehmen an, das Objekt liegt beim ersten Pick

        # --- 3. PLANUNGSSCHLEIFE ---
        for i, target_entry in enumerate(targets):
            
            # --- PARSING DER ZIELE (NEU) ---
            # Prüfen, ob das Format (Koordinate, Aktion) ist
            current_goal_coords = target_entry
            current_action = None
            
            # Heuristik: Ist es ein Tupel/Liste der Länge 2 und ist das zweite Element ein String?
            if isinstance(target_entry, (tuple, list)) and len(target_entry) == 2 and isinstance(target_entry[1], str):
                current_goal_coords = target_entry[0]
                current_action = target_entry[1]
            else:
                # Altes Format (nur Koordinaten)
                current_goal_coords = target_entry
                current_action = "MOVE" # Default

            # Planner erwartet goalList als Liste von Koordinaten
            current_goal_list_for_planner = [current_goal_coords] 
            
            print(f"        [Runner] Segment {i}: Action={current_action}")
            
            # 1. Planen
            segment = planner.planPath(current_start, current_goal_list_for_planner, config)
            
            if segment is None or len(segment) == 0:
                print(f"        No path found in segment {i}")
                break
            
            # 2. Graph & Pfad verarbeiten (Standard Logic)
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
                current_segment_start_node = relabeled_segment[0]
                full_roadmap_graph.add_edge(last_segment_end_node, current_segment_start_node, connection="virtual")

            if not full_relabeled_path:
                full_relabeled_path.extend(relabeled_segment)
            else:
                full_relabeled_path.extend(relabeled_segment)

            # --- ACTION HANDLING ---
            # Wo im Gesamtpfad sind wir JETZT (am Ende dieses Segments)?
            current_end_index = len(full_relabeled_path) - 1

            if current_action == "PICK":
                # Wir merken uns: Bei Index X passiert ein PICK
                action_events[current_end_index] = ("PICK", object_shape)

                # A) Statisches Hindernis entfernen (falls vorhanden)
                if current_world_obstacle_poly is not None:
                    if current_world_obstacle_poly in planner._collisionChecker.obstacles:
                        planner._collisionChecker.obstacles.remove(current_world_obstacle_poly)
                        print(f"      [Action] PICK: Removed static object from obstacles.")
                    current_world_obstacle_poly = None # Referenz löschen

                # B) Objekt an Roboter hängen
                if object_shape is not None:
                    print(f"      [Action] PICK: Robot holds object.")
                    planner._collisionChecker.attach_object(object_shape)
            
            elif current_action == "PLACE":
                # Wir merken uns: Bei Index X passiert ein PLACE
                action_events[current_end_index] = ("PLACE", None)

                if hasattr(planner._collisionChecker, 'detach_object'):
                    # 1. Position des Objekts berechnen, BEVOR wir es loslassen
                    # Wir brauchen die Konfiguration, an der der Roboter gerade steht (das Ziel dieses Segments)
                    # current_goal_list_for_planner ist eine Liste [config], wir nehmen das erste Element
                    place_config = current_goal_list_for_planner[0]

                    # Wir stellen sicher, dass das Objekt 'gegriffen' ist für die Berechnung
                    # (Falls es im Step davor schon dran war, ist das redundant, aber sicher ist sicher)
                    if hasattr(planner._collisionChecker, 'attached_object_shape') and planner._collisionChecker.attached_object_shape is None:
                         # Fallback: Falls wir PLACE rufen, aber er hat nix (sollte nicht passieren)
                         pass
                    
                    # Geometrie berechnen
                    geo = planner._collisionChecker.get_robot_geometry(place_config)
                    placed_object_poly = geo.get('held_object')

                    # 2. Das Objekt als NEUES HINDERNIS hinzufügen
                    if placed_object_poly is not None:
                        print(f"      [Action] PLACE: Object added to obstacles at current position.")
                        planner._collisionChecker.obstacles.append(placed_object_poly)
                        current_world_obstacle_poly = placed_object_poly # Referenz merken (falls wir es später wieder aufheben)

                    print(f"        [Action] PLACE executed. Robot is empty.")
                    planner._collisionChecker.detach_object()

            # Setup für nächste Runde
            last_segment_end_node = relabeled_segment[-1]
            current_start = current_goal_list_for_planner

        # --- FINALE MANIPULATION ---
        if full_relabeled_path:
            real_start_id = full_relabeled_path[0]
            if real_start_id in full_roadmap_graph.nodes:
                start_pos = full_roadmap_graph.nodes[real_start_id]['pos']
                full_roadmap_graph.add_node("start", pos=start_pos) 

            real_goal_id = full_relabeled_path[-1]
            if real_goal_id in full_roadmap_graph.nodes:
                goal_pos = full_roadmap_graph.nodes[real_goal_id]['pos']
                full_roadmap_graph.add_node("goal", pos=goal_pos) 

        planner.graph = full_roadmap_graph
        planner.collidingEdges = full_colliding_edges         
        planner.nonCollidingEdges = full_non_colliding_edges  

        # Cleanup am Ende (wichtig für Plots)
        if hasattr(planner._collisionChecker, 'detach_object'):
            planner._collisionChecker.detach_object()
        # Auch das temporäre Hindernis entfernen, damit der Checker "sauber" ist für den nächsten Run
        if current_world_obstacle_poly in planner._collisionChecker.obstacles:
            planner._collisionChecker.obstacles.remove(current_world_obstacle_poly)
        
        return full_relabeled_path, action_events