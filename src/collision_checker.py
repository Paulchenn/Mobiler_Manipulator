import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point
from shapely.affinity import translate, rotate

class CollisionChecker:
    """
    Collision Checker for a planar mobile manipulator.
    Fully compatible with HKA IP-Planners (LazyPRM, VisibilityPRM).
    Now supports Pick & Place (Attach/Detach Objects).
    """

    def __init__(self, base_shape, arm_config, gripper_config, gripper_len=0.0, arm_base_offset=(0, 0), base_center=None, object_shape=None, limits=None, intersect_limit=0.05, check_self_collision_flag=True):
        self.base_shape_def = base_shape
        self.arm_config = arm_config
        self.gripper_config=gripper_config,
        self.gripper_config=list(self.gripper_config)[0]
        self.gripper_length=gripper_len,
        self.gripper_length=list(self.gripper_length)[0]
        self.arm_base_offset = arm_base_offset
        if base_center is None:
            print(f"[CollisionChecker]: No base center set - using (0,0)")
            self.base_center = (0,0)
        else:
            self.base_center = base_center
        self.obstacles = []
        self.check_self_collision_flag = check_self_collision_flag
        self.intersect_limit = intersect_limit
        
        # Default Limits
        # print(len(limits))
        # print(self.getDim())
        if len(limits)==self.getDim():
            self.limits = limits
        else:
            # print(f"[CollisionChecker]: Warning - Limits dimension {len(limits)} does not match robot DoF {self.getDim()}. Using no limits.")
            raise ValueError(f"[CollisionChecker]: Limits dimension {len(limits)} does not match robot DoF {self.getDim()}")
            
        # Variable storing the shape of attached object (as polygon centered at origin)
        # The origin (0,0) represents the gripper contact point
        self.attached_object_shape = None 

        self.object_shape = object_shape

    # --- Pick & Place Interface ---

    def attach_object(self, object_polygon_points):
        """
        Attach an object to the end effector (gripper).
        
        Args:
            object_polygon_points: List of (x, y) coordinate tuples defining object geometry.
                                   Object should be centered at origin (0, 0), which represents
                                   the gripper contact point where the object is grasped.
        """
        self.attached_object_shape = Polygon(object_polygon_points)
        # print("[CollisionChecker] Object attached.")

    def detach_object(self):
        """Remove object from end effector (release from gripper)."""
        self.attached_object_shape = None
        # print("[CollisionChecker] Object detached.")

    # --- HKA Planner Interface Requirements ---

    def getDim(self):
        return 3 + len(self.arm_config)

    def getEnvironmentLimits(self):
        return self.limits
    
    def set_sampling_limits(self, limits):
        self.limits = limits

    def get_object_shape(self):
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
        
        # Add held object to collision check list if attached
        if geo['held_object'] is not None:
            robot_parts.append(geo['held_object'])
            robot_parts_wGripper.append(geo['held_object'])

        # 3. Check geometric workspace limits
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

                # 1. Check against base
                if obj.intersects(base):
                     if obj.intersection(base).area > self.intersect_limit:
                        return True
                     
                # 2. Check against arm segments
                # Object is attached to last arm segment, so check against all segments except the last
                last_segment_index = len(arm_segments) - 1

                for i, seg in enumerate(arm_segments):
                    # Skip the segment where object is attached (would always cause collision)
                    if i == last_segment_index:
                        continue
                        
                    if obj.intersects(seg):
                        # Use intersect_limit for robustness
                        if obj.intersection(seg).area > self.intersect_limit:
                            return True

        return False

    def lineInCollision(self, config1, config2, step_size=0.2):
        """Check if a linear path between two configurations is collision-free.
        
        Args:
            config1: Starting configuration
            config2: Ending configuration
            step_size: Distance between interpolation points along the path
            
        Returns:
            True if path collides, False if collision-free
        """
        p1 = np.array(config1)
        p2 = np.array(config2)
        dist = np.linalg.norm(p2 - p1)
        if dist == 0: return self.pointInCollision(p1)
        n_steps = int(dist / step_size) + 1
        for i in range(n_steps + 1):
            alpha = i / n_steps
            config_interp = p1 * (1 - alpha) + p2 * alpha
            if self.pointInCollision(config_interp):
                return True
        return False

    # --- Robot Geometry Computation ---

    def set_obstacles(self, obstacle_list):
        """Set static obstacle polygons in the environment.
        
        Args:
            obstacle_list: List of obstacle geometries, each as a list of (x, y) coordinate tuples
        """
        self.obstacles = [Polygon(obs) for obs in obstacle_list]

    def get_robot_geometry(self, config):
        """Compute collision geometry for robot at given configuration.
        
        Args:
            config: Configuration [x, y, theta, q1, q2, ...] where:
                   - x, y: base position in world frame
                   - theta: base orientation
                   - q1, q2, ...: joint angles
                   
        Returns:
            Dictionary with polygon geometries:
                - base: base polygon in world frame
                - arm_segments: list of arm segment polygons
                - gripper: gripper polygon (None if not configured)
                - held_object: attached object polygon (None if not attached)
        """
        x, y, theta = config[0:3]
        joint_angles = config[3:]

        # 1. Compute base geometry in world frame
        base_poly = Polygon(self.base_shape_def)
        base_poly = rotate(base_poly, theta, origin=(self.base_center), use_radians=True)
        base_poly = translate(base_poly, xoff=x-self.base_center[0], yoff=y-self.base_center[1])

        # 2. Compute arm geometry by forward kinematics
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
            
            # Create cylinder representation of arm segment
            line = LineString([(current_x, current_y), (end_x, end_y)])
            segment_poly = line.buffer(width / 2.0)
            arm_polys.append(segment_poly)
            current_x, current_y = end_x, end_y
                
        # ---------------------------------------------------------
        # Gripper Calculation
        # ---------------------------------------------------------
        gripper_poly = None
        
        # Tool Center Point (TCP): where object contact occurs
        # Defaults to end of last arm segment
        tcp_x = current_x
        tcp_y = current_y
        
        if self.gripper_config is not None:
            # 1. Create gripper polygon (defined around origin)
            gripper_poly = Polygon(self.gripper_config)
            
            # 2. Rotate gripper with last arm segment
            gripper_poly = rotate(gripper_poly, current_angle, origin=(0,0), use_radians=True)
            
            # 3. Translate to arm end position
            gripper_poly = translate(gripper_poly, xoff=current_x, yoff=current_y)
            
            # 4. Compute TCP (Tool Center Point)
            # Extend TCP along current end-effector direction by gripper_length
            if self.gripper_length > 0:
                tcp_x = current_x + self.gripper_length * np.cos(current_angle)
                tcp_y = current_y + self.gripper_length * np.sin(current_angle)

        # ---------------------------------------------------------
        # Held Object Calculation
        # ---------------------------------------------------------
        held_obj_poly = None
        if self.attached_object_shape is not None:
            obj = self.attached_object_shape
            
            # Rotate object with gripper/end-effector orientation
            obj = rotate(obj, current_angle, origin=(0,0), use_radians=True)
            
            # Translate object to TCP (gripper tip), not arm end
            # This positions the object at the actual grasp point
            obj = translate(obj, xoff=tcp_x, yoff=tcp_y)
            
            held_obj_poly = obj

        return {
            "base": base_poly, 
            "arm_segments": arm_polys, 
            "gripper": gripper_poly, 
            "held_object": held_obj_poly
        }
        

        
    def drawObstacles(self, ax):
        """Visualize static obstacles on the given matplotlib axes.
        
        Args:
            ax: Matplotlib axes object for drawing
        """
        for obs in self.obstacles:
            x, y = obs.exterior.xy
            ax.fill(x, y, fc='gray', alpha=0.5, ec='black')

    def draw(self, config, ax=None):
        """Visualize robot and obstacles at given configuration.
        
        Args:
            config: Robot configuration [x, y, theta, q1, q2, ...]
            ax: Matplotlib axes object (creates new figure if None)
        """
        if ax is None: fig, ax = plt.subplots()
        self.drawObstacles(ax)  # Reuse obstacle visualization
        self.drawRobot(config, ax)
        ax.set_aspect('equal')

    def drawRobot(self, config, ax, alpha=0.3, color='blue', action='MOVE'):
        if action == 'PICK':
            if self.object_shape is not None:
                self.attach_object(self.object_shape)
            else:
                print("        [Warning] PICK action requested, but no object shape defined or unavailable.")
        elif action == "PLACE":
            self.detach_object()
        elif action == 'MOVE':
            pass
        else:
            print(f"        [Action] Action '{action}' is not recognized.")

        geo = self.get_robot_geometry(config)
        
        # 1. Draw base
        bx, by = geo['base'].exterior.xy
        ax.fill(bx, by, fc=color, alpha=alpha, ec='black', linewidth=1)
        
        # 2. Draw arm segments
        i=1
        for seg in geo['arm_segments']:
            i+=1
            sx, sy = seg.exterior.xy
            ax.fill(sx, sy, fc='orange', alpha=alpha, ec='black', linewidth=1)

        if geo['gripper'] is not None:
            gx, gy = geo['gripper'].exterior.xy
            ax.fill(gx, gy, fc='#333333', alpha=0.9, ec='black', linewidth=1, label="Gripper")

        # Draw held object if attached
        if geo['held_object'] is not None:
            ox, oy = geo['held_object'].exterior.xy
            # Draw object in green for visibility
            ax.fill(ox, oy, fc='#00FF00', alpha=0.9, ec='black', linewidth=1, label="Held Object")