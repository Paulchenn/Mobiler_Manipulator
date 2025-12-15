# coding: utf-8
"""
Benchmark Suite for Mobile Manipulator (5-DOF).
"""

import numpy as np
from shapely.geometry import Polygon, LineString, Point

# Importiere DEINEN CollisionChecker
# Wir gehen davon aus, dass diese Datei im selben Ordner wie collision_checker.py liegt
try:
    from collision_checker import CollisionChecker
except ImportError:
    from src.collision_checker import CollisionChecker

# Import Benchmark Class
from src.planners.IPBenchmark import Benchmark

# --------------------------------------------------------------------------------
# SELF CHECK ON/OFF
# --------------------------------------------------------------------------------
SELF_CHECK = True
INTERSECT_LIMITS = 0.002

# --------------------------------------------------------------------------------
# START GOAL DEFINITION
# --------------------------------------------------------------------------------
START = [[-3.0, -4.0, 0.0, np.pi/4, -np.pi/4]]
GOAL  = [[2.0, 3.0, 0.0, np.pi/2, -np.pi/4], [3.0, -4.0, 0.0, 0.0, -np.pi/2]]

# --------------------------------------------------------------------------------
# LIMITS DEFINITION
# --------------------------------------------------------------------------------
LIMITS = [-5, 5]

# --------------------------------------------------------------------------------
# ROBOTER DEFINITION (Zentralisiert)
# --------------------------------------------------------------------------------
# Diese Definition wird für alle Benchmarks verwendet
ROBOT_BASE_SHAPE = [(0, 0), (2, 0), (2, 1), (0, 1)]
ROBOT_CENTER = [1,0.5]
ROBOT_ARM_CONFIG = [
    [0.5, 0.1, [0, 3.14]], # Gelenk 1
    [1.0, 0.1, [-3.14, 3.14]]  # Gelenk 2
]
# Arm startet leicht versetzt vorne an der Basis
ARM_OFFSET = (2, 1)

def create_checker(obstacles):
    """Hilfsfunktion, um einen Checker mit Hindernissen zu erzeugen"""
    cc = CollisionChecker(base_shape=ROBOT_BASE_SHAPE, arm_config=ROBOT_ARM_CONFIG, arm_base_offset=ARM_OFFSET, base_center=ROBOT_CENTER, limits=LIMITS, check_self_collision_flag=SELF_CHECK, intersect_limit=INTERSECT_LIMITS)
    cc.set_obstacles(obstacles)
    # print(cc)
    return cc

benchList = list()

# --------------------------------------------------------------------------------
# Benchmark 1: Empty World (Sanity Check)
# --------------------------------------------------------------------------------
desc_1 = "Freier Raum. Testet, ob der Planer eine direkte Linie findet."
cc_1 = create_checker([]) # Keine Hindernisse

# Start: (0,0), Arm gestreckt
start_1 = [[0.0, 0.0, 0.0, 0.0, 0.0]]
# Ziel: (10,10), Arm eingeklappt
goal_1  = [[10.0, 10.0, 1.57, 1.5, -1.5]]

# --------------------------------------------------------------------------------
# Benchmark 2: The Wall (Doorway)
# --------------------------------------------------------------------------------
# Eine Wand bei X=5 mit einer Lücke bei Y=0
wall_lower = Polygon([(-0.2, LIMITS[0]), (0.2, LIMITS[0]), (0.2, -1), (-0.2, -1)])
wall_upper = Polygon([(-0.2, 1), (0.2, 1), (0.2, LIMITS[1]), (-0.2, LIMITS[1])])
obstacles_2 = [wall_upper, wall_lower]

desc_2 = "Eine Wand mit einer Lücke. Der Roboter muss durchfahren."
cc_2 = create_checker(obstacles_2)

start_2 = [[0.0, 0.0, 0.0, 0.0, 0.0]]
goal_2  = [[10.0, 0.0, 0.0, 0.0, 0.0]]

# --------------------------------------------------------------------------------
# Benchmark 3: Narrow Passage (Schwer)
# --------------------------------------------------------------------------------
# Ein sehr enger Gang. Basis ist 2m breit (von -1 bis 1). Lücke ist 3m breit.
obs_left  = Polygon([(LIMITS[0], -2), (-1, -2), (-1, 2), (LIMITS[0], 2)])
obs_right = Polygon([(1, -2), (LIMITS[1], -2), (LIMITS[1], 2), (1, 2)])
obstacles_3 = [obs_left, obs_right]

desc_3 = "Narrow Passage. Erfordert präzise Basis-Bewegung."
cc_3 = create_checker(obstacles_3)

start_3 = [[0.0, 0.0, 0.0, 0.0, 0.0]]
goal_3  = [[12.0, 0.0, 0.0, 0.0, 0.0]]

# --------------------------------------------------------------------------------
# Benchmark 4: Cluttered Forest
# --------------------------------------------------------------------------------
# Viele kleine Boxen
obstacles_4 = []
positions = [(-4,4), (0,4),
             (-4,0), (0,0), (4,0),
                     (0,-4)
             ]
for (px, py) in positions:
    obstacles_4.append(Point(px, py).buffer(0.4)) # Runde Säulen

desc_4 = "Clutter. Roboter muss slalomen, Arm darf nicht ausschlagen."
cc_4 = create_checker(obstacles_4)

start_4 = [[0.0, 0.0, 0.0, 0.0, 0.0]]
goal_4  = [[12.0, 0.0, 0.0, 0.0, 0.0]]

# --------------------------------------------------------------------------------
# Benchmark 5: Shelf Reach (Arm Test)
# --------------------------------------------------------------------------------
# Ein Hindernis blockiert den direkten Weg, Roboter muss "um die Ecke" greifen
# Basis kann nicht zum Ziel (blockiert), Arm muss arbeiten.
barrier = LineString([(0,2), (4, 2), (4, 3.7)]).buffer(0.2, cap_style='flat') # Wand vor dem Ziel
obstacles_5 = [barrier]

desc_5 = "Reach Task. Die Basis kann das Ziel nicht erreichen, der Arm muss rüberreichen."
cc_5 = create_checker(obstacles_5)

start_5 = [[0.0, 0.0, 0.0, 0.0, 0.0]]
# Ziel ist HINTER der Wand (bei X=8), Arm muss lang gemacht werden
# Basis bleibt vor der Wand stehen (z.B. X=3)
goal_5  = [[4.0, 2.0, 0.0, 0.5, 0.5]] 

# --------------------------------------------------------------------------------
# Append Benchmarks
# --------------------------------------------------------------------------------
benchList.append(Benchmark("Empty World",       cc_1, START, GOAL, desc_1))
benchList.append(Benchmark("The Wall",          cc_2, START, GOAL, desc_2))
benchList.append(Benchmark("Narrow Passage",    cc_3, START, GOAL, desc_3))
benchList.append(Benchmark("Forest",            cc_4, START, GOAL, desc_4))
benchList.append(Benchmark("Shelf Reach",       cc_5, START, GOAL, desc_5))