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
        if limits!=self.getDim():
            # self.limits_x = limits[0]
            # self.limits_y = limits[1]
            # self.limits_theta = limits[2]
            # self.limits_1 = limits[3][0]
            # self.limits_2 = limits[3][1]
            self.limits = limits
            
        # print(f"[CollisionChecker]: self.limits: {self.limits}; self.check_self_collision_flag: {self.check_self_collision_flag}")

        # ### NEU: Variable für das angehängte Objekt ###
        # Speichert die Form des Objekts (als Polygon, zentriert um 0,0)
        self.attached_object_shape = None 

        self.object_shape = object_shape

    # --- Pick & Place Interface (NEU) ---

    def attach_object(self, object_polygon_points):
        """
        Hängt ein Objekt an den End-Effector.
        Args:
            object_polygon_points: Liste von (x,y) Punkten, die das Objekt definieren.
                                   Das Objekt sollte um (0,0) definiert sein.
                                   (0,0) ist der Punkt, an dem der Greifer zugreift.
        """
        self.attached_object_shape = Polygon(object_polygon_points)
        # print("[CollisionChecker] Object attached.")

    def detach_object(self):
        """Entfernt das Objekt vom End-Effector."""
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
        """Checks if a config is valid (Collision, Joint Limits & Workspace Boundaries)."""
        
        # 1. Check Joint Limits
        joint_angles = config[2:]
        for i, segment in enumerate(self.limits[2:]):
            if not (segment[0] <= joint_angles[i] <= segment[1]):
                return True
            
        # 2. Geometry Calculation
        geo = self.get_robot_geometry(config)
        robot_parts = [geo['base']] + geo['arm_segments']
        robot_parts_wGripper = robot_parts + [geo['gripper']] if geo['gripper'] is not None else robot_parts
        
        # ### NEU: Objekt zur Liste der zu prüfenden Teile hinzufügen ###
        if geo['held_object'] is not None:
            robot_parts.append(geo['held_object'])
            robot_parts_wGripper.append(geo['held_object'])

        # 3. Geometric workspace limits check
        if self.limits is not None:
            x_lim = self.limits[0]
            y_lim = self.limits[1]
            
            for part in robot_parts_wGripper:
                minx, miny, maxx, maxy = part.bounds
                if minx < x_lim[0] or maxx > x_lim[1]:
                    return True
                if miny < y_lim[0] or maxy > y_lim[1]:
                    return True

        # 4. Obstacles check
        for part in robot_parts_wGripper:
            for obs in self.obstacles:
                if part.intersects(obs):
                    return True

        # 5. Self Collision
        if self.check_self_collision_flag:
            base = geo['base']
            arm_segments = geo['arm_segments']
            gripper = geo['gripper']

            # A) Wir prüfen Armsegmente gegen Basis und Gripper
            for seg in arm_segments:
                if seg.intersection(base).area > self.intersect_limit:
                    return True
                elif seg.intersects(gripper):
                    if seg != arm_segments[-1]:
                        return True
                    
            # B) Wir prüfen Gripper gegen Basis:
            if gripper.intersects(base):
                return True
                    
            # B) Wir prüfen das getragene Objekt (falls vorhanden)
            if geo['held_object'] is not None:
                obj = geo['held_object']

                # 1. Gegen die Basis
                if obj.intersects(base):
                     if obj.intersection(base).area > self.intersect_limit:
                        return True
                     
                # 2. Gegen die Armsegmente (NEU!)
                # Das Objekt hängt am letzten Segment (Index = len - 1).
                # Wir prüfen gegen alle Segmente AUSSER dem letzten.
                last_segment_index = len(arm_segments) - 1

                for i, seg in enumerate(arm_segments):
                    # Überspringe das Segment, an dem das Objekt hängt (führt sonst immer zu Kollision)
                    if i == last_segment_index:
                        continue
                        
                    if obj.intersects(seg):
                        # Wir nutzen auch hier intersect_limit für Robustheit
                        if obj.intersection(seg).area > self.intersect_limit:
                            return True

        return False

    def lineInCollision(self, config1, config2, step_size=0.2):
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
            base_poly = Polygon(self.base_shape_def)
            base_poly = rotate(base_poly, theta, origin=(self.base_center), use_radians=True)
            base_poly = translate(base_poly, xoff=x-self.base_center[0], yoff=y-self.base_center[1])

            # 2. Arm Geometry Calculation
            arm_polys = []
            ox, oy = self.arm_base_offset
            cx, cy = self.base_center
            
            dx = ox - cx
            dy = oy - cy

            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            rotated_dx = dx * cos_t - dy * sin_t
            rotated_dy = dx * sin_t + dy * cos_t

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
                
            # ---------------------------------------------------------
            # ### NEU: Gripper Calculation ###
            # ---------------------------------------------------------
            gripper_poly = None
            
            # Variable für den Anfass-Punkt des Objekts (TCP)
            # Standardmäßig ist es das Ende des Arms
            tcp_x = current_x
            tcp_y = current_y
            
            if self.gripper_config is not None:
                # 1. Gripper Polygon erstellen (definiert um 0,0)
                gripper_poly = Polygon(self.gripper_config)
                
                # 2. Rotieren (der Gripper dreht sich mit dem letzten Armsegment mit)
                gripper_poly = rotate(gripper_poly, current_angle, origin=(0,0), use_radians=True)
                
                # 3. Verschieben an das Ende des Arms
                gripper_poly = translate(gripper_poly, xoff=current_x, yoff=current_y)
                
                # 4. TCP BERECHNEN (Hier passiert die Magie!)
                # Wir schieben den Punkt, an dem das Objekt hängen soll, 
                # um gripper_length in Richtung des aktuellen Winkels weiter.
                if self.gripper_length > 0:
                    tcp_x = current_x + self.gripper_length * np.cos(current_angle)
                    tcp_y = current_y + self.gripper_length * np.sin(current_angle)

            # ---------------------------------------------------------
            # ### Held Object Calculation ###
            # ---------------------------------------------------------
            held_obj_poly = None
            if self.attached_object_shape is not None:
                obj = self.attached_object_shape
                
                # Objekt rotieren (es dreht sich mit dem Gripper)
                obj = rotate(obj, current_angle, origin=(0,0), use_radians=True)
                
                # WICHTIG: Jetzt verschieben wir es an den TCP (Gripper-Spitze), 
                # nicht mehr an current_x (Arm-Ende)
                obj = translate(obj, xoff=tcp_x, yoff=tcp_y)
                
                held_obj_poly = obj

            return {
                "base": base_poly, 
                "arm_segments": arm_polys, 
                "gripper": gripper_poly, 
                "held_object": held_obj_poly
            }
        
    def drawObstacles(self, ax):
        for obs in self.obstacles:
            x, y = obs.exterior.xy
            ax.fill(x, y, fc='gray', alpha=0.5, ec='black')

    def draw(self, config, ax=None):
        if ax is None: fig, ax = plt.subplots()
        self.drawObstacles(ax) # Code Reuse
        self.drawRobot(config, ax)
        ax.set_aspect('equal')

    def drawRobot(self, config, ax, alpha=0.3, color='blue', action='MOVE'):
        if action == 'PICK':
            if self.object_shape is not None:
                self.attach_object(self.object_shape)
            else:
                print("        [Warnung] Aktion PICK angefordert, aber kein ObjektShape oder keine Fähigkeit gefunden.")
        elif action == "PLACE":
            # print(f"        [Action] PLACE executed. Robot is empty.")
            self.detach_object()
        elif action == 'MOVE':
            pass
        else:
            print(f"        [Action] Action {action} not know!!!")

        # print(config)
        geo = self.get_robot_geometry(config)
        
        # 1. Basis
        bx, by = geo['base'].exterior.xy
        ax.fill(bx, by, fc=color, alpha=alpha, ec='black', linewidth=1)
        
        # 2. Arm
        for seg in geo['arm_segments']:
            sx, sy = seg.exterior.xy
            ax.fill(sx, sy, fc='orange', alpha=alpha, ec='black', linewidth=1)

        if geo['gripper'] is not None:
            gx, gy = geo['gripper'].exterior.xy
            ax.fill(gx, gy, fc='#333333', alpha=0.9, ec='black', linewidth=1, label="Gripper")

        # ### NEU: Objekt zeichnen ###
        if geo['held_object'] is not None:
            ox, oy = geo['held_object'].exterior.xy
            # Wir zeichnen das Objekt in Grün (oder einer anderen Farbe), damit es auffällt
            ax.fill(ox, oy, fc='#00FF00', alpha=0.9, ec='black', linewidth=1, label="Held Object")