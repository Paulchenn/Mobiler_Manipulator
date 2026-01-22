# coding: utf-8

"""
Benchmark scenario definition module for path planning evaluation.

This module defines benchmark test cases used to evaluate motion planning algorithms.
Based on 'Introduction to robot path planning' course (Author: Bjoern Hein).

License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
See: http://creativecommons.org/licenses/by-nc/4.0/
"""


class Benchmark:
    """
    Defines a path planning benchmark scenario for algorithm evaluation.

    A benchmark encapsulates a robot scenario with initial configurations,
    goal configurations, collision environment, and task description.
    Supports both basic point-to-point navigation and multi-goal Pick & Place tasks.
    """

    def __init__(self, name, collisionChecker, startList, goalList, description):
        """
        Initialize a benchmark scenario.

        Args:
            name (str): Identifier for the benchmark scenario
            collisionChecker (CollisionChecker): Environment definition with obstacle
                geometry and collision detection
            startList (list): Robot starting configuration(s). List format depends on
                robot DOF. Example: [x, y, theta] for a 3-DOF mobile manipulator
            goalList (list): Sequence of goal configurations or (config, action) tuples.
                Supports:
                - List of configurations: [config1, config2] (MOVE actions only)
                - List of tuples: [(config1, "PICK"), (config2, "PLACE")] (with actions)
            description (str): Human-readable description of the benchmark scenario,
                including task specification and expected difficulty

        Attributes:
            name (str): Benchmark name
            collisionChecker (CollisionChecker): Collision detection environment
            startList (list): Starting configuration(s)
            goalList (list): Goal configuration(s) and actions
            description (str): Scenario description
        """
        self.name = name
        self.collisionChecker = collisionChecker
        self.startList = startList
        self.goalList = goalList
        self.description = description