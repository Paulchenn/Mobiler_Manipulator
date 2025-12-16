# Datei: MultiGoalPlannerRunner.py
import networkx as nx

class MultiGoalPlannerRunner:
    """
    Führt einen Pfadplaner sequenziell für mehrere Ziele aus (Start -> Ziel1 -> Ziel2 ...)
    und fügt die Ergebnisse (Graphen, Pfade, Kanten) zu einer Gesamtlösung zusammen,
    damit diese korrekt visualisiert werden kann.
    """

    @staticmethod
    def run_benchmark(planner, benchmark, config):
        """
        Führt die Planung durch und modifiziert das planner-Objekt in-place,
        indem es den Gesamtgraphen unterschiebt.
        
        Returns:
            full_relabeled_path (list): Der komplette Pfad über alle Stationen.
        Raises:
            Exception: Wenn ein Teilsegment nicht geplant werden kann.
        """
        
        # Sammel-Container für den gesamten Lauf
        full_relabeled_path = []
        full_roadmap_graph = nx.Graph()
        full_colliding_edges = []       
        full_non_colliding_edges = []   
        
        current_start = benchmark.startList 
        targets = benchmark.goalList
        last_segment_end_node = None
        
        # --- EXTERNE SCHLEIFE FÜR MULTI-GOAL ---
        for i, target in enumerate(targets):
            current_goal = [target] 
            
            # 1. Planen (Planner löscht intern seinen Graphen meistens)
            segment = planner.planPath(current_start, current_goal, config)
            
            if segment is None or len(segment) == 0:
                print(f"        No path found in segment {i}")# (from {current_start} to {current_goal})")
                break
            
            # 2. Mapping erstellen (Alt -> Neu)
            # Wir prefixen JEDEN Knoten mit dem Segment-Index (s0_, s1_, ...)
            current_graph = planner.graph
            mapping = {node: f"s{i}_{node}" for node in current_graph.nodes()}
            
            # 3. Pfad übersetzen
            relabeled_segment = [mapping[n] for n in segment]
            
            # 4. Graphen umbenennen und mergen
            relabeled_graph = nx.relabel_nodes(current_graph, mapping)
            full_roadmap_graph = nx.compose(full_roadmap_graph, relabeled_graph)
            
            # 5. KANTEN ÜBERSETZEN UND SAMMELN
            if hasattr(planner, 'collidingEdges'):
                for u, v in planner.collidingEdges:
                    if u in mapping and v in mapping:
                        full_colliding_edges.append((mapping[u], mapping[v]))
                        
            if hasattr(planner, 'nonCollidingEdges'):
                for u, v in planner.nonCollidingEdges:
                    if u in mapping and v in mapping:
                        full_non_colliding_edges.append((mapping[u], mapping[v]))

            # 6. Visuelle Verbindung zwischen Segmenten
            if last_segment_end_node is not None:
                current_segment_start_node = relabeled_segment[0]
                # Virtuelle Kante, damit der Pfad optisch nicht unterbrochen ist
                full_roadmap_graph.add_edge(last_segment_end_node, current_segment_start_node, connection="virtual") 

            # 7. Pfad verlängern
            if not full_relabeled_path:
                full_relabeled_path.extend(relabeled_segment)
            else:
                # Wir fügen den KOMPLETTEN neuen Pfad an (inkl. Startknoten des neuen Segments),
                # da wir ja eine virtuelle Kante dorthin gezogen haben.
                full_relabeled_path.extend(relabeled_segment)

            # Setup für nächste Runde
            last_segment_end_node = relabeled_segment[-1]
            current_start = current_goal

        # --- FINALE MANIPULATION ---
        
        # TRICK: Fake Start/Goal Knoten hinzufügen für Visualisierung
        if full_relabeled_path:
            # Echter Start (z.B. s0_start)
            real_start_id = full_relabeled_path[0]
            if real_start_id in full_roadmap_graph.nodes:
                start_pos = full_roadmap_graph.nodes[real_start_id]['pos']
                full_roadmap_graph.add_node("start", pos=start_pos) 

            # Echtes Ziel (z.B. s3_goal)
            real_goal_id = full_relabeled_path[-1]
            if real_goal_id in full_roadmap_graph.nodes:
                goal_pos = full_roadmap_graph.nodes[real_goal_id]['pos']
                full_roadmap_graph.add_node("goal", pos=goal_pos) 

        # Dem Planner ALLES unterschieben
        planner.graph = full_roadmap_graph
        planner.collidingEdges = full_colliding_edges         
        planner.nonCollidingEdges = full_non_colliding_edges  
        
        return full_relabeled_path