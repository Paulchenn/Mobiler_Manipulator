# coding: utf-8

"""
This code is part of the course 'Innovative Programmiermethoden für Industrieroboter' (Author: Bjoern Hein). It is based on the slides given during the course, so please **read the information in theses slides first**

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

import networkx as nx
def visibilityPRMVisualize(planner, solution, actions, ax = None, nodeSize = 300, plot_only_solution=True, plot_robot=True):
    # get a list of positions of all nodes by returning the content of the attribute 'pos'
    graph = planner.graph
    statsHandler = planner.statsHandler
    collChecker = planner._collisionChecker
    pos = nx.get_node_attributes(graph,'pos')
    pos_xy  = {k: v[:2] for k, v in pos.items()}
    color = nx.get_node_attributes(graph,'color')
    # print(dir(planner))
    object_shape = getattr(planner, 'objectShape', None)

    collChecker.drawObstacles(ax)
    
    if not plot_only_solution: #statsHandler:
        statPos = nx.get_node_attributes(statsHandler.graph,'pos')
        statPos_xy  = {k: v[:2] for k, v in statPos.items()}
        nx.draw_networkx_nodes(statsHandler.graph, pos=statPos_xy, ax=ax, alpha=0.5, node_size=nodeSize)
        nx.draw_networkx_edges(statsHandler.graph, pos=statPos_xy, ax=ax, alpha=0.1, width=0.2)#, edge_color='y')
        
    # draw graph 
    if not plot_only_solution:
        nx.draw_networkx_nodes(graph, pos_xy, ax = ax, nodelist=list(color.keys()), node_color=list(color.values()), node_size=nodeSize, alpha=0.5)
        nx.draw_networkx_edges(graph, pos_xy, ax = ax, alpha=0.1, width=0.2)

    # how largest connected component
    if not plot_only_solution and solution!=[]:
        Gcc = sorted(nx.connected_components(graph), key=len, reverse=True)
        G0=graph.subgraph(Gcc[0])# = largest connected component
        # nx.draw_networkx_edges(G0, pos_xy, edge_color='b', width=2.0, ax = ax)
        pass
    
    # get nodes based on solution path
    Gsp = nx.subgraph(graph,solution)
        
    # draw start and goal
    if "start" in graph.nodes(): 
        nx.draw_networkx_nodes(graph, pos_xy, nodelist=["start"], node_size=nodeSize, node_color='#00dd00',  ax=ax)
        nx.draw_networkx_labels(graph,pos_xy,labels={"start": "S"},  ax=ax)
    if "goal" in graph.nodes():
        nx.draw_networkx_nodes(graph, pos_xy, nodelist=["goal"], node_size=nodeSize, node_color='#dd0000',  ax=ax)
        nx.draw_networkx_labels(graph, pos_xy, labels={"goal": "G"},  ax=ax)

    # --- LÖSUNGSPFAD ---
    if solution != []:
        Gsp = nx.subgraph(graph, solution)
        # print("-"*20)
        # print(actions[2])
        # print("-"*20)
        # draw nodes based on solution path
        nx.draw_networkx_nodes(Gsp, pos_xy, ax=ax, alpha=1, node_size=nodeSize)
        # draw edges based on solution path
        nx.draw_networkx_edges(Gsp, pos_xy, ax=ax, alpha=1, edge_color='g', width=3, label="Solution Path") # War width=10
        
        # --- NEU: Roboter entlang des Pfades zeichnen ---
        # Wir iterieren über jeden Knoten im Lösungspfad
        for i, node_id in enumerate(solution):
            try:
                if i == 0:
                    collChecker.detach_object()
                    action = 'MOVE'
                else:
                    action = actions[i-1][0]
            except:
                action = 'MOVE'
            # print("-"*20)
            # print(action)
            # print("-"*20)

            # 1. Volle 5D-Konfiguration holen
            full_config = graph.nodes[node_id]['pos']
            # print(full_config)
            
            # 2. Styling: Start und Ziel deckend, dazwischen transparent
            if i == 0 or i == len(solution) - 1:
                current_alpha = 1
            else:
                current_alpha = 0.5 # Sehr transparent, damit man den Graphen noch sieht
            
            # 3. Roboter zeichnen lassen
            if plot_robot:
                collChecker.drawRobot(full_config, ax, alpha=current_alpha, action=action)
    
    return

