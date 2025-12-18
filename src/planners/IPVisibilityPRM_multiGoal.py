# coding: utf-8

"""
This code is part of the course "Introduction to robot path planning" (Author: Bjoern Hein).

License is based on Creative Commons: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) (pls. check: http://creativecommons.org/licenses/by-nc/4.0/)
"""

from IPPRMBase import PRMBase
import networkx as nx
from scipy.spatial import cKDTree
from IPPerfMonitor import IPPerfMonitor

class VisibilityStatsHandler():
    
    def __init__(self):
        self.graph = nx.Graph()
        
    def addNodeAtPos(self,nodeNumber,pos):
        self.graph.add_node(nodeNumber, pos=pos, color='yellow')
        return
    
    def addVisTest(self,fr,to):
        self.graph.add_edge(fr, to)
        return
        
class VisPRM(PRMBase):
    """Class implements an simplified version of a visibility PRM"""

    def __init__(self, _collChecker, _statsHandler = None):
        super(VisPRM, self).__init__(_collChecker)
        self.graph = nx.Graph()
        self.statsHandler = VisibilityStatsHandler() # not yet fully customizable (s. parameters of constructors)
                
    def _isVisible(self, pos, guardPos):
        return not self._collisionChecker.lineInCollision(pos, guardPos)

    @IPPerfMonitor
    def _learnRoadmap(self, ntry):

        nodeNumber = 0
        currTry = 0
        while currTry < ntry:
            #print currTry
            # select a random  free position
            q_pos = self._getRandomFreePosition()
            if self.statsHandler:
                self.statsHandler.addNodeAtPos(nodeNumber, q_pos)
           
            g_vis = None
        
            # every connected component represents one guard
            merged = False
            for comp in nx.connected_components(self.graph): # Impliciteley represents G_vis
                found = False
                merged = False
                for g in comp: # connected components consists of guards and connection: only test nodes of type 'Guards'
                    if self.graph.nodes()[g]['nodeType'] == 'Guard':
                        if self.statsHandler:
                            self.statsHandler.addVisTest(nodeNumber, g)
                        if self._isVisible(q_pos,self.graph.nodes()[g]['pos']):
                            found = True
                            if g_vis == None:
                                g_vis = g
                            else:
                                self.graph.add_node(nodeNumber, pos = q_pos, color='lightblue', nodeType = 'Connection')
                                self.graph.add_edge(nodeNumber, g)
                                self.graph.add_edge(nodeNumber, g_vis)
                                merged = True
                        # break, if node was visible,because visibility from one node of the guard is sufficient...
                        if found == True: break;
                # break, if connection was found. Reason: computed connected components (comp) are not correct any more, 
                # they've changed because of merging
                if merged == True: # how  does it change the behaviour? What has to be done to keep the original behaviour?
                    break;                    

            if (merged==False) and (g_vis == None):
                self.graph.add_node(nodeNumber, pos = q_pos, color='red', nodeType = 'Guard')
                #print "ADDED Guard ", nodeNumber
                currTry = 0
            else:
                currTry += 1

            nodeNumber += 1

    @IPPerfMonitor
    def planPath(self, startList, goalList, config):
        """
        
        Args:
            start (array): start position in planning space
            goal (array) : goal position in planning space
            config (dict): dictionary with the needed information about the configuration options
            
        Example:
        
            config["ntry"] = 40 
        
        """
        # 0. reset
        self.graph.clear()
        
        # 1. check start and goal whether collision free (s. BaseClass)
        checkedStartList, checkedGoalList = self._checkStartGoal(startList,goalList)
        
        # 2. learn Roadmap
        self._learnRoadmap(config["ntry"])

        # 3. find connection of start and goal to roadmap
        # find nearest, collision-free connection between node on graph and start
        posList = nx.get_node_attributes(self.graph,'pos')
        # kdTree = cKDTree(list(posList.values()))

        # Helper: Um Knoten-IDs aus dem KDTree Index zurückzuholen
        # (Wichtig, da posList.keys() nicht zwingend sortiert ist)
        roadmapNodeKeys = list(posList.keys()) 
        roadmapPositions = [posList[k] for k in roadmapNodeKeys]
        
        if not roadmapPositions:
            # Falls Roadmap leer ist (kann bei ntry=0 oder sehr engem Raum passieren)
            return []
        
        kdTree = cKDTree(roadmapPositions)        
        
        # result = kdTree.query(checkedStartList[0],k=5)
        # for node in result[1]:
        #     if not self._collisionChecker.lineInCollision(checkedStartList[0],self.graph.nodes()[list(posList.keys())[node]]['pos']):
        #          self.graph.add_node("start", pos=checkedStartList[0], color='lightgreen')
        #          self.graph.add_edge("start", list(posList.keys())[node])
        #          break
            
        # --- A) Connect Start ---
        # (Wir nehmen an, es gibt nur einen Startpunkt, sonst müsste man hier auch loopen)
        startPos = checkedStartList[0]
        result = kdTree.query(startPos, k=5)
        
        start_connected = False
        for node_idx in result[1]:
            # Index-Check, falls k größer als Anzahl Knoten
            if node_idx >= len(roadmapNodeKeys): continue
                
            targetNodeID = roadmapNodeKeys[node_idx]
            targetPos = self.graph.nodes()[targetNodeID]['pos']
            
            if not self._collisionChecker.lineInCollision(startPos, targetPos):
                 self.graph.add_node("start", pos=startPos, color='lightgreen')
                 self.graph.add_edge("start", targetNodeID)
                 start_connected = True
                 break
        
        if not start_connected:
            return [] # Start konnte nicht verbunden werden

        # result = kdTree.query(checkedGoalList[0],k=5)
        # for node in result[1]:
        #     if not self._collisionChecker.lineInCollision(checkedGoalList[0],self.graph.nodes()[list(posList.keys())[node]]['pos']):
        #          self.graph.add_node("goal", pos=checkedGoalList[0], color='lightgreen')
        #          self.graph.add_edge("goal", list(posList.keys())[node])
        #          break

        # --- B) Connect All Goals ---
        goal_ids = []
        for i, goalPos in enumerate(checkedGoalList):
            goal_id = "goal_{}".format(i)
            goal_ids.append(goal_id)
            
            result = kdTree.query(goalPos, k=5)
            goal_connected = False
            
            for node_idx in result[1]:
                if node_idx >= len(roadmapNodeKeys): continue
                    
                targetNodeID = roadmapNodeKeys[node_idx]
                targetPos = self.graph.nodes()[targetNodeID]['pos']
                
                if not self._collisionChecker.lineInCollision(goalPos, targetPos):
                     self.graph.add_node(goal_id, pos=goalPos, color='lightgreen')
                     self.graph.add_edge(goal_id, targetNodeID)
                     goal_connected = True
                     break
            
            if not goal_connected:
                print(f"Warnung: {goal_id} konnte nicht mit Roadmap verbunden werden.")
                return []

        # try:
        #     path = nx.shortest_path(self.graph,"start","goal")
        # except:
        #     return []
        # print(path)
        # return path
        
        # 4. Search sequential path
        # start -> goal_0 -> goal_1 ...
        full_path = []
        current_start_node = "start"

        try:
            for next_goal_id in goal_ids:
                # Pfadsegment berechnen
                segment = nx.shortest_path(self.graph, current_start_node, next_goal_id)
                
                # Zusammenkleben:
                # Wenn full_path schon existiert, schneiden wir das erste Element des neuen Segments ab,
                # da es identisch mit dem letzten Element des vorherigen Segments ist.
                if not full_path:
                    full_path.extend(segment)
                else:
                    full_path.extend(segment[1:])
                
                current_start_node = next_goal_id
                
        except nx.NetworkXNoPath:
            # Ein Segment konnte nicht gefunden werden
            return []

        # print(full_path)
        return full_path
