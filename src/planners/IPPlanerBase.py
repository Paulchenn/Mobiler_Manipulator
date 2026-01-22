# coding: utf-8

"""
Base class for motion planning algorithms.

This module provides the abstract base class for all motion planning algorithms.
Handles common functionality like collision checking, path validation, and
start/goal configuration filtering.

Based on 'Introduction to robot path planning' course (Author: Bjoern Hein).
License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
See: http://creativecommons.org/licenses/by-nc/4.0/
"""


class PlanerBase:
    """
    Abstract base class for motion planning algorithms.

    Provides common interface and utilities for path planning implementations.
    Handles configuration space validation, collision checking, and start/goal
    filtering.
    """

    def __init__(self, collisionChecker):
        """
        Initialize the base planner.

        Args:
            collisionChecker (CollisionChecker): Environment instance for collision
                detection and robot geometry validation
        """
        self._collisionChecker = collisionChecker

    def _checkStartGoal(self, startList, goalList):
        """
        Validate and filter start and goal configurations.

        Checks that configurations are:
        1. Correct dimension for the robot (matches degrees of freedom)
        2. Not in collision with obstacles

        Args:
            startList (list): List of candidate start configurations
            goalList (list): List of candidate goal configurations

        Returns:
            tuple: (validStartList, validGoalList) - Lists of valid configurations

        Raises:
            Exception: If no valid start or goal configurations remain after filtering
        """
        newStartList = list()
        for start in startList:
            # Check dimension compatibility
            if len(start) != self._collisionChecker.getDim():
                continue
            # Check collision-free
            if self._collisionChecker.pointInCollision(start):
                continue
            newStartList.append(start)

        newGoalList = list()
        for goal in goalList:
            # Check dimension compatibility
            if len(goal) != self._collisionChecker.getDim():
                print(f"len(goal) != self._collisionChecker.getDim(): {len(goal)} != {self._collisionChecker.getDim()})")
                continue
            # Check collision-free
            if self._collisionChecker.pointInCollision(goal):
                print(f"self._collisionChecker.pointInCollision(goal): {self._collisionChecker.pointInCollision(goal)}")
                continue
            newGoalList.append(goal)


        return newStartList, newGoalList