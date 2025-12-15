import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString
from shapely.affinity import translate, rotate

class CollisionChecker:
    """
    Collision Checker for a planar mobile manipulator.
    Fully compatible with HKA IP-Planners (LazyPRM, VisibilityPRM).
    """

    def __init__(self, base_shape, arm_config, arm_base_offset=(0, 0), base_center=None, limits=[0.0, 22.0], intersect_limit=0.05, check_self_collision_flag=True):
        self.base_shape_def = base_shape
        self.arm_config = arm_config
        self.arm_base_offset = arm_base_offset
        if base_center is None:
            print(f"[CollisionChecker]: No base center set - using (0,0)")
            self.base_center = (0,0)
        else:
            # print(f"[CollisionChecker]: Base center set - using {base_center}")
            self.base_center = base_center
        self.obstacles = []
        self.check_self_collision_flag = check_self_collision_flag

        self.intersect_limit = intersect_limit
        
        # Default Limits (will be overwritten from Notebook)
        # Format: [[min, max], [min, max], ...] for 5 Dimensions
        self.limits = [limits] * self.getDim()
        print(f"[CollisionChecker]: self.limits: {self.limits}; self.check_self_collision_flag: {self.check_self_collision_flag}")

    # --- HKA Planner Interface Requirements ---

    def getDim(self):
        """Returns the dimension of the configuration space (x, y, theta, q1, q2...)."""
        return 3 + len(self.arm_config)

    def getEnvironmentLimits(self):
        """Returns the sampling limits for the planner."""
        return self.limits
    
    def set_sampling_limits(self, limits):
        """Helper to set limits from the notebook."""
        self.limits = limits

    def pointInCollision(self, config):
        """Checks if a config is valid (Collision, Joint Limits & Workspace Boundaries)."""
        
        # 1. Check Joint Limits (Arm specific) - fastest check first
        joint_angles = config[3:]
        for i, segment in enumerate(self.arm_config):
            limits = segment[2]
            if not (limits[0] <= joint_angles[i] <= limits[1]):
                return True
            
        # 2. Geometry Calculation (for all further checks needed)
        geo = self.get_robot_geometry(config)
        robot_parts = [geo['base']] + geo['arm_segments']

        # 3. Geometric workspace limits check
        # Checks whether any part of the robot (bounding box) leaves the boundaries.
        if self.limits is not None:
            # We assume that self.limits[0] = X limits and self.limits[1] = Y limits.
            x_lim = self.limits[0]
            y_lim = self.limits[1]
            
            for part in robot_parts:
                # part.bounds gibt (minx, miny, maxx, maxy) zurück
                minx, miny, maxx, maxy = part.bounds
                
                # Check X
                if minx < x_lim[0] or maxx > x_lim[1]:
                    return True
                # Check Y
                if miny < y_lim[0] or maxy > y_lim[1]:
                    return True

        # 3. Obstacles check
        for part in robot_parts:
            for obs in self.obstacles:
                if part.intersects(obs):
                    return True

        # 4. Self Collision
        def print_inter_areas(inter_areas):
            print(f"Intersection areas (limit={self.intersect_limit}):")
            for i, inter_area in enumerate(inter_areas):
                if i == 0:
                    print(f"    base  -> arm_{i+1}: {inter_area:.4f}")
                else:
                    print(f"    arm_{i} -> arm_{i+1}: {inter_area:.4f}")
        if self.check_self_collision_flag:
            base = geo['base']
            inter_areas = []
            for seg in geo['arm_segments']:
                if seg.intersects(base):
                    # Ignore touching contact at joint (area ~ 0)
                    inter_areas.append(seg.intersection(base).area)
                    if seg.intersection(base).area > self.intersect_limit:
                        # print_inter_areas(inter_areas)
                        return True
            # print_inter_areas(inter_areas)
        return False

    def lineInCollision(self, config1, config2, step_size=0.2):
        """Checks if the direct path between two configs is collision-free."""
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

    # --- Internal Geometry Logic ---

    def set_obstacles(self, obstacle_list):
        self.obstacles = [Polygon(obs) for obs in obstacle_list]

    def get_robot_geometry(self, config):
        x, y, theta = config[0:3]
        joint_angles = config[3:]

        # 1. Base Geometry
        # Here you use shapely ‘rotate’ with your custom origin.
        # print(self.base_center)
        base_poly = Polygon(self.base_shape_def)
        base_poly = rotate(base_poly, theta, origin=(self.base_center), use_radians=True)
        base_poly = translate(base_poly, xoff=x-self.base_center[0], yoff=y-self.base_center[1])

        # 2. Arm Geometry Calculation
        arm_polys = []

        # Get local coordinates
        ox, oy = self.arm_base_offset   # Where is the arm attached?
        cx, cy = self.base_center       # What does the robot revolve around?
        
        # Calculate vector from center of rotation to arm starting point
        dx = ox - cx
        dy = oy - cy

        # Prepare rotation
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Rotate the vector (dx, dy)
        rotated_dx = dx * cos_t - dy * sin_t
        rotated_dy = dx * sin_t + dy * cos_t

        # Calculate new starting position:
        # World position (x, y) + Local center of rotation (cx, cy) + Rotated vector
        # (Since Shapely's ‘translate’ moves the polygon by (x,y), we must do the same here)
        current_x = x + cx + rotated_dx - self.base_center[0]
        current_y = y + cy + rotated_dy - self.base_center[1]
        current_angle = theta

        for i, segment in enumerate(self.arm_config):
            length, width = segment[0], segment[1]
            q = joint_angles[i]
            current_angle += q
            
            end_x = current_x + length * np.cos(current_angle)
            end_y = current_y + length * np.sin(current_angle)
            
            line = LineString([(current_x, current_y), (end_x, end_y)])
            segment_poly = line.buffer(width / 2.0)
            arm_polys.append(segment_poly)
            current_x, current_y = end_x, end_y
            
        return {"base": base_poly, "arm_segments": arm_polys}
    
    def drawObstacles(self, ax):
        """
        Draws only the obstacles (required by IPVIS visualization).
        """
        for obs in self.obstacles:
            x, y = obs.exterior.xy
            ax.fill(x, y, fc='gray', alpha=0.5, ec='black')

    def draw(self, config, ax=None):
        if ax is None: fig, ax = plt.subplots()
        for obs in self.obstacles:
            x, y = obs.exterior.xy
            ax.fill(x, y, alpha=0.5, fc='gray', ec='black')
        geo = self.get_robot_geometry(config)
        bx, by = geo['base'].exterior.xy
        ax.fill(bx, by, alpha=0.7, fc='blue', ec='black')
        for i, seg in enumerate(geo['arm_segments']):
            sx, sy = seg.exterior.xy
            ax.fill(sx, sy, alpha=0.7, fc='orange', ec='black')
        ax.set_aspect('equal')

    def drawRobot(self, config, ax, alpha=0.3, color='blue'):
        """
        Zeichnet nur den Roboter an der gegebenen Konfiguration.
        Wird von den IPVIS-Funktionen genutzt, um den Pfad darzustellen.
        """
        geo = self.get_robot_geometry(config)
        
        # 1. Basis zeichnen
        bx, by = geo['base'].exterior.xy
        # Wir nutzen 'fc' (facecolor) für Füllung und 'ec' (edgecolor) für den Rand
        ax.fill(bx, by, fc=color, alpha=alpha, ec='black', linewidth=1)
        
        # 2. Arm zeichnen
        for seg in geo['arm_segments']:
            sx, sy = seg.exterior.xy
            # Arm machen wir immer orange-ish, oder passen es an
            ax.fill(sx, sy, fc='orange', alpha=alpha, ec='black', linewidth=1)