"""
Collision checking and robot geometry module for mobile manipulators.

Provides collision detection, workspace validation, and forward kinematics
for planar mobile manipulators (mobile base + articulated arm + gripper).
Supports dynamic object attachment/detachment for Pick & Place tasks.

Fully compatible with HKA IP-Planners (LazyPRM, VisibilityPRM, RRT, AStar, etc.).
"""

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point
from shapely.affinity import translate, rotate


class CollisionChecker:
    """
    Collision detection and robot geometry computation for planar mobile manipulators.

    Provides methods for:
    - Collision detection (point and line segment checking)
    - Forward kinematics and robot geometry computation
    - Pick & Place manipulation (object attachment/detachment)
    - Workspace validation and visualization
    - Configuration space joint limit checking

    The robot model consists of:
    - Mobile base: Rectangular platform with configurable shape
    - Articulated arm: Series of revolute joints with link geometry
    - Gripper: End-effector for grasping (optional)
    - Attached objects: Dynamically attached to gripper for Pick & Place

    Configuration space: [x, y, theta, q1, q2, ...] where:
        - x, y: Base position in world frame (meters)
        - theta: Base orientation (radians)
        - q1, q2, ...: Joint angles (radians)
    """

    def __init__(self, base_shape, arm_config, gripper_config, gripper_len=0.0,
                 arm_base_offset=(0, 0), base_center=None, object_shape=None,
                 limits=None, intersect_limit=0.05, check_self_collision_flag=True):
        """
        Initialize the collision checker with robot geometry.

        Args:
            base_shape (list): List of (x, y) tuples defining base polygon
            arm_config (list): List of (length, width) tuples for arm segments
            gripper_config (list): List of (x, y) tuples defining gripper polygon
            gripper_len (float): Length of gripper/TCP offset (meters)
            arm_base_offset (tuple): Offset of arm base from mobile base center (x, y)
            base_center (tuple): Center point of base polygon for rotation
            object_shape (list): List of (x, y) tuples defining object geometry
            limits (list): Configuration space limits [(x_min, x_max), (y_min, y_max),
                          (theta_min, theta_max), (q1_min, q1_max), ...]
            intersect_limit (float): Minimum intersection area (m²) to flag collision
            check_self_collision_flag (bool): Enable self-collision checking
        """
        self.base_shape_def = base_shape
        self.arm_config = arm_config
        self.gripper_config = gripper_config
        self.gripper_length = gripper_len
        self.arm_base_offset = arm_base_offset
        
        if base_center is None:
            print(f"[CollisionChecker]: No base center set - using (0,0)")
            self.base_center = (0, 0)
        else:
            self.base_center = base_center
        
        self.obstacles = []
        self.check_self_collision_flag = check_self_collision_flag
        self.intersect_limit = intersect_limit
        
        # Configuration space dimension validation
        if len(limits) == self.getDim():
            self.limits = limits
        else:
            raise ValueError(f"[CollisionChecker]: Limits dimension {len(limits)} does not match robot DoF {self.getDim()}")
        
        # Variable storing the shape of attached object (centered at origin)
        # Origin (0,0) represents the gripper contact point where object is grasped
        self.attached_object_shape = None
        self.object_shape = object_shape

    # --- Pick & Place Interface ---

    def attach_object(self, object_polygon_points):
        """
        Attach an object to the end effector (gripper).

        Stores object geometry for collision checking during manipulation tasks.
        Object vertices should be specified relative to the gripper contact point.

        Args:
            object_polygon_points (list): List of (x, y) coordinate tuples defining
                object geometry. The origin (0, 0) represents the gripper contact point
                where the object is grasped.
        """
        self.attached_object_shape = Polygon(object_polygon_points)

    def detach_object(self):
        """
        Release object from end effector (detach from gripper).

        Called after PLACE actions to indicate the gripper no longer holds an object.
        """
        self.attached_object_shape = None

    # --- Planner Interface Requirements ---

    def getDim(self):
        """
        Get configuration space dimension (degrees of freedom).

        Returns:
            int: DOF = 3 (base x, y, theta) + number of arm joints
        """
        return 3 + len(self.arm_config)

    def getEnvironmentLimits(self):
        """
        Get configuration space limits.

        Returns:
            list: Limits for each DOF as [(min, max), ...]
        """
        return self.limits
    
    def set_sampling_limits(self, limits):
        """
        Update configuration space limits.

        Args:
            limits (list): New limits for each DOF as [(min, max), ...]
        """
        self.limits = limits

    def get_object_shape(self):
        """
        Get the object shape definition.

        Returns:
            list: Object geometry vertices or None if no object defined
        """
        return self.object_shape

    def pointInCollision(self, config):
        """Check if a configuration is valid (joint limits, workspace bounds, collisions)."""
        
        # 1. Check joint limits
        joint_angles = config[2:]
        for i, segment in enumerate(self.limits[2:]):
            if not (segment[0] <= joint_angles[i] <= segment[1]):
                return True
            
        # 2. Compute robot geometry at configuration
        geo = self.get_robot_geometry(config)
        robot_parts = [geo['base']] + geo['arm_segments']
        robot_parts_wGripper = robot_parts + [geo['gripper']] if geo['gripper'] is not None else robot_parts
        
        # Add held object to collision check if attached
        if geo['held_object'] is not None:
            robot_parts.append(geo['held_object'])
            robot_parts_wGripper.append(geo['held_object'])

        # 3. Check workspace limits
        if self.limits is not None:
            x_lim = self.limits[0]
            y_lim = self.limits[1]
            
            for part in robot_parts_wGripper:
                minx, miny, maxx, maxy = part.bounds
                if minx < x_lim[0] or maxx > x_lim[1]:
                    return True
                if miny < y_lim[0] or maxy > y_lim[1]:
                    return True

        # 4. Check collisions with static obstacles
        for part in robot_parts_wGripper:
            for obs in self.obstacles:
                if part.intersects(obs):
                    return True

        # 5. Check self-collisions
        if self.check_self_collision_flag:
            base = geo['base']
            arm_segments = geo['arm_segments']
            gripper = geo['gripper']

            # Check arm segments against base and gripper
            for seg in arm_segments:
                if seg.intersection(base).area > self.intersect_limit:
                    return True
                elif seg.intersects(gripper):
                    if seg != arm_segments[-1]:
                        return True
                    
            # Check gripper against base
            if gripper.intersects(base):
                return True
                    
            # Check held object against robot parts (if attached)
            if geo['held_object'] is not None:
                obj = geo['held_object']

                # Check against base
                if obj.intersects(base):
                    if obj.intersection(base).area > self.intersect_limit:
                        return True
                     
                # Check against arm segments
                # Object is attached to last arm segment, so check against all others
                last_segment_index = len(arm_segments) - 1

                for i, seg in enumerate(arm_segments):
                    # Skip last segment where object is attached
                    if i == last_segment_index:
                        continue
                        
                    if obj.intersects(seg):
                        # Use intersect_limit for robustness
                        if obj.intersection(seg).area > self.intersect_limit:
                            return True

        return False

    def lineInCollision(self, config1, config2, step_size=0.2):
        """
        Check if a linear straight-line path between two configurations is collision-free.

        Interpolates between configurations at regular intervals and checks each
        interpolation point for collision.

        Args:
            config1 (array-like): Starting configuration
            config2 (array-like): Ending configuration
            step_size (float): Distance between interpolation points along path

        Returns:
            bool: True if path contains collision, False if completely collision-free
        """
        p1 = np.array(config1)
        p2 = np.array(config2)
        dist = np.linalg.norm(p2 - p1)
        if dist == 0: 
            return self.pointInCollision(p1)
        n_steps = int(dist / step_size) + 1
        for i in range(n_steps + 1):
            alpha = i / n_steps
            config_interp = p1 * (1 - alpha) + p2 * alpha
            if self.pointInCollision(config_interp):
                return True
        return False

    # --- Robot Geometry Computation ---

    def set_obstacles(self, obstacle_list):
        """
        Set static obstacle polygons in the environment.

        Args:
            obstacle_list (list): List of obstacles, each defined as a list of
                (x, y) coordinate tuples forming the polygon vertices
        """
        self.obstacles = [Polygon(obs) for obs in obstacle_list]

    def get_robot_geometry(self, config):
        """
        Compute collision geometry for the robot at a given configuration.

        Uses forward kinematics to compute positions of all robot parts (base, arm,
        gripper, held objects) and returns their polygon representations for
        collision checking.

        Args:
            config (array-like): Configuration [x, y, theta, q1, q2, ...] where:
                - x, y: Base position in world frame (meters)
                - theta: Base orientation (radians)
                - q1, q2, ...: Joint angles (radians)

        Returns:
            dict: Robot geometry with keys:
                - base (Polygon): Base polygon in world frame
                - arm_segments (list): List of arm segment polygons
                - gripper (Polygon): Gripper geometry (None if not configured)
                - held_object (Polygon): Attached object polygon (None if not attached)
        """
        x, y, theta = config[0:3]
        joint_angles = config[3:]

        # 1. Compute base geometry in world frame
        base_poly = Polygon(self.base_shape_def)
        base_poly = rotate(base_poly, theta, origin=(self.base_center), use_radians=True)
        base_poly = translate(base_poly, xoff=x - self.base_center[0], yoff=y - self.base_center[1])

        # 2. Compute arm geometry using forward kinematics
        arm_polys = []
        ox, oy = self.arm_base_offset
        cx, cy = self.base_center
        
        # Compute arm base offset in world frame
        dx = ox - cx
        dy = oy - cy

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Rotate arm base offset by base orientation
        rotated_dx = dx * cos_t - dy * sin_t
        rotated_dy = dx * sin_t + dy * cos_t

        # Starting position for forward kinematics
        current_x = x + cx + rotated_dx - self.base_center[0]
        current_y = y + cy + rotated_dy - self.base_center[1]
        current_angle = theta

        # Build arm segments using forward kinematics
        for i, segment in enumerate(self.arm_config):
            length, width = segment[0], segment[1]
            q = joint_angles[i]
            current_angle += q
            
            # Compute segment end point
            end_x = current_x + length * np.cos(current_angle)
            end_y = current_y + length * np.sin(current_angle)
            
            # Create cylinder representation of arm segment as buffered line
            line = LineString([(current_x, current_y), (end_x, end_y)])
            segment_poly = line.buffer(width / 2.0)
            arm_polys.append(segment_poly)
            current_x, current_y = end_x, end_y
                
        # --- Gripper Calculation ---
        gripper_poly = None
        
        # Tool Center Point (TCP): where object contact occurs
        # Defaults to end of last arm segment
        tcp_x = current_x
        tcp_y = current_y
        
        if self.gripper_config is not None:
            # Create gripper polygon (defined around origin in local frame)
            gripper_poly = Polygon(self.gripper_config)
            
            # Rotate gripper with last arm segment orientation
            gripper_poly = rotate(gripper_poly, current_angle, origin=(0, 0), use_radians=True)
            
            # Translate gripper to arm end position
            gripper_poly = translate(gripper_poly, xoff=current_x, yoff=current_y)
            
            # Compute TCP (Tool Center Point) - extends beyond gripper
            if self.gripper_length > 0:
                tcp_x = current_x + self.gripper_length * np.cos(current_angle)
                tcp_y = current_y + self.gripper_length * np.sin(current_angle)

        # --- Held Object Calculation ---
        held_obj_poly = None
        if self.attached_object_shape is not None:
            obj = self.attached_object_shape
            
            # Rotate object with gripper/end-effector orientation
            obj = rotate(obj, current_angle, origin=(0, 0), use_radians=True)
            
            # Translate object to TCP (gripper tip), not arm segment end
            # This positions object at the actual grasp point
            obj = translate(obj, xoff=tcp_x, yoff=tcp_y)
            
            held_obj_poly = obj

        return {
            "base": base_poly,
            "arm_segments": arm_polys,
            "gripper": gripper_poly,
            "held_object": held_obj_poly
        }
    def drawObstacles(self, ax):
        """
        Visualize static obstacles on matplotlib axes.

        Args:
            ax (matplotlib.axes.Axes): Matplotlib axes object for drawing
        """
        for obs in self.obstacles:
            x, y = obs.exterior.xy
            ax.fill(x, y, fc='gray', alpha=0.5, ec='black')

    def draw(self, config, ax=None):
        """
        Visualize robot and obstacles at a given configuration.

        Args:
            config (array-like): Robot configuration [x, y, theta, q1, q2, ...]
            ax (matplotlib.axes.Axes, optional): Matplotlib axes (creates new if None)
        """
        if ax is None:
            fig, ax = plt.subplots()
        self.drawObstacles(ax)
        self.drawRobot(config, ax)
        ax.set_aspect('equal')

    def drawRobot(self, config, ax, alpha=0.3, color='blue', action='MOVE'):
        """
        Draw robot at a given configuration including attached objects.

        Args:
            config (array-like): Robot configuration [x, y, theta, q1, q2, ...]
            ax (matplotlib.axes.Axes): Matplotlib axes for drawing
            alpha (float): Transparency level (0-1)
            color (str): Base color for robot (default 'blue')
            action (str): Current action ('MOVE', 'PICK', 'PLACE')
        """
        if action == 'PICK':
            if self.object_shape is not None:
                self.attach_object(self.object_shape)
            else:
                print("        [Warning] PICK action requested, but no object shape defined.")
        elif action == "PLACE":
            self.detach_object()
        elif action == 'MOVE':
            pass
        else:
            print(f"        [Action] Action '{action}' is not recognized.")

        geo = self.get_robot_geometry(config)
        
        # 1. Draw mobile base
        bx, by = geo['base'].exterior.xy
        ax.fill(bx, by, fc=color, alpha=alpha, ec='black', linewidth=1)
        
        # 2. Draw arm segments
        for seg in geo['arm_segments']:
            sx, sy = seg.exterior.xy
            ax.fill(sx, sy, fc='orange', alpha=alpha, ec='black', linewidth=1)

        # 3. Draw gripper
        if geo['gripper'] is not None:
            gx, gy = geo['gripper'].exterior.xy
            ax.fill(gx, gy, fc='#333333', alpha=0.9, ec='black', linewidth=1, label="Gripper")

        # 4. Draw held object if attached
        if geo['held_object'] is not None:
            ox, oy = geo['held_object'].exterior.xy
            ax.fill(ox, oy, fc='#00FF00', alpha=0.9, ec='black', linewidth=1, label="Held Object")