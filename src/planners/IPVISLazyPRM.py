# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein). It is based on the slides given during the course, so please **read the information in theses slides first**

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import networkx as nx


def lazyPRMVisualize(planner, solution=[], ax=None, nodeSize=300, plot_only_solution=True):
    graph = planner.graph.copy()
    collChecker = planner._collisionChecker
    collEdges = planner.collidingEdges
    nonCollEdges = planner.nonCollidingEdges
    
    # 1. Positionen extrahieren und auf 2D (x,y) projizieren
    # Das ist wichtig für die 5D->2D Darstellung
    pos = nx.get_node_attributes(graph, 'pos')
    pos_xy = {k: v[:2] for k, v in pos.items()}
    
    # 2. Farben holen (falls vorhanden)
    color_dict = nx.get_node_attributes(graph, 'color')

    # Hindernisse zeichnen
    collChecker.drawObstacles(ax)

    # --- KNOTEN ZEICHNEN (Korrigiert) ---
    # Wir erstellen eine Farbliste für ALLE Knoten. 
    # Wenn ein Knoten keine Farbe hat, nehmen wir Standard-Blau ('#1f78b4').
    all_nodes = list(graph.nodes())
    node_colors = [color_dict.get(n, '#1f78b4') for n in all_nodes]
    
    if not plot_only_solution:
        nx.draw_networkx_nodes(graph, pos_xy, ax=ax, 
                           nodelist=all_nodes, 
                           node_color=node_colors, 
                           node_size=nodeSize)

    # --- KANTEN ZEICHNEN ---
    # Alle Kanten (Basis in schwarz)
    if not plot_only_solution:
        nx.draw_networkx_edges(graph, pos_xy, ax=ax)

    # Größte verbundene Komponente hervorheben (blau gestrichelt)
    try:
        Gcc = (graph.subgraph(c) for c in nx.connected_components(graph))
        G0 = next(Gcc) 
        if not plot_only_solution:
            nx.draw_networkx_edges(G0, pos_xy, ax=ax, edge_color='b', width=3.0, style='dashed', alpha=0.5)
    except StopIteration:
        pass # Graph ist leer

    # Kollidierende Kanten (Rot)
    if collEdges:
        collGraph = nx.Graph()
        collGraph.add_nodes_from(graph.nodes(data=True))
        for i in collEdges:
            collGraph.add_edge(i[0], i[1])
        if not plot_only_solution:
            nx.draw_networkx_edges(collGraph, pos_xy, ax=ax, alpha=0.2, edge_color='r', width=5)

    # Freie Kanten (Gelb)
    if nonCollEdges:
        nonCollGraph = nx.Graph()
        nonCollGraph.add_nodes_from(graph.nodes(data=True))
        for i in nonCollEdges:
            nonCollGraph.add_edge(i[0], i[1])
        if not plot_only_solution:
            nx.draw_networkx_edges(nonCollGraph, pos_xy, ax=ax, alpha=0.8, edge_color='gold', width=5)
    
    # --- START / ZIEL ---
    if "start" in graph.nodes(): 
        nx.draw_networkx_nodes(graph, pos_xy, ax=ax, nodelist=["start"], node_size=nodeSize*1.2, node_color='#00dd00')
        nx.draw_networkx_labels(graph, pos_xy, ax=ax, labels={"start": "S"})

    if "goal" in graph.nodes():
        nx.draw_networkx_nodes(graph, pos_xy, ax=ax, nodelist=["goal"], node_size=nodeSize*1.2, node_color='#dd0000')
        nx.draw_networkx_labels(graph, pos_xy, ax=ax, labels={"goal": "G"})

    # --- LÖSUNGSPFAD ---
    if solution != []:
        # draw nodes based on solution path
        Gsp = nx.subgraph(graph, solution)
        # draw edges based on solution path
        nx.draw_networkx_nodes(Gsp, pos_xy, ax=ax, alpha=0.8, node_size=nodeSize)
        nx.draw_networkx_edges(Gsp, pos_xy, ax=ax, alpha=0.8, edge_color='g', width=8) # War width=10
        
        # --- NEU: Roboter entlang des Pfades zeichnen ---
        # Wir iterieren über jeden Knoten im Lösungspfad
        for i, node_id in enumerate(solution):
            # 1. Volle 5D-Konfiguration holen
            full_config = graph.nodes[node_id]['pos']
            
            # 2. Styling: Start und Ziel deckend, dazwischen transparent
            if i == 0 or i == len(solution) - 1:
                current_alpha = 0.8
            else:
                current_alpha = 0.15 # Sehr transparent, damit man den Graphen noch sieht
            
            # 3. Roboter zeichnen lassen
            collChecker.drawRobot(full_config, ax, alpha=current_alpha)
    
    return

