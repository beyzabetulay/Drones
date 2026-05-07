import heapq
import random
from enum import Enum

import numpy as np
from vispy import app, scene
from vispy.scene import visuals

from kalman import Kalman


WORLD_SIZE = 100
NUM_OBSTACLES = 75
MAX_RENDER_BLOCKS = 8000

DT = 0.03
MAX_SPEED = 3.0

SENSOR_SIGMA_POS = 0.8
SENSOR_SIGMA_VEL = 0.3

RANDOM_SEED = None


class Status(Enum):
    SUCCESS = 1
    FAILURE = 2
    RUNNING = 3


class BlockWorld:
    def __init__(self, size):
        self.size = size
        self.grid = np.zeros((size, size, size), dtype=bool)

    def inside(self, x, y, z):
        return 0 <= x < self.size and 0 <= y < self.size and 0 <= z < self.size

    def is_occupied(self, p):
        x, y, z = p
        if not self.inside(x, y, z):
            return True
        return bool(self.grid[x, y, z])

    def is_free(self, p):
        return not self.is_occupied(p)

    def add_box(self, x, y, z, w, d, h):
        x2 = min(x + w, self.size)
        y2 = min(y + d, self.size)
        z2 = min(z + h, self.size)

        if x2 <= x or y2 <= y or z2 <= z:
            return

        self.grid[x:x2, y:y2, z:z2] = True

    def add_sphere_blob(self, cx, cy, cz, r):
        r2 = r * r

        for x in range(cx - r, cx + r + 1):
            for y in range(cy - r, cy + r + 1):
                for z in range(cz - r, cz + r + 1):
                    if not self.inside(x, y, z):
                        continue

                    dx = x - cx
                    dy = y - cy
                    dz = z - cz

                    if dx * dx + dy * dy + dz * dz <= r2:
                        self.grid[x, y, z] = True

    def add_l_shape(self, x, y, z, length, thickness, height):
        self.add_box(x, y, z, length, thickness, height)
        self.add_box(x, y, z, thickness, length, height)

    def add_tower(self, x, y, z, width, height):
        self.add_box(x, y, z, width, width, height)

    def generate_random_obstacles(self, count):
        for _ in range(count):
            obstacle_type = random.choice(["box", "sphere", "l_shape", "tower"])

            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            z = random.randint(0, self.size - 1)

            if obstacle_type == "box":
                self.add_box(
                    x, y, z,
                    random.randint(3, 12),
                    random.randint(3, 12),
                    random.randint(3, 14),
                )

            elif obstacle_type == "sphere":
                self.add_sphere_blob(
                    x, y, z,
                    random.randint(3, 8),
                )

            elif obstacle_type == "l_shape":
                self.add_l_shape(
                    x, y, z,
                    random.randint(6, 16),
                    random.randint(2, 4),
                    random.randint(3, 12),
                )

            elif obstacle_type == "tower":
                self.add_tower(
                    x, y, z,
                    random.randint(3, 6),
                    random.randint(10, 35),
                )

    def random_free_cell(self):
        while True:
            p = (
                random.randint(0, self.size - 1),
                random.randint(0, self.size - 1),
                random.randint(0, self.size - 1),
            )
            if self.is_free(p):
                return p

    def occupied_coordinates(self):
        return np.argwhere(self.grid)

    def print_summary(self):
        occupied = int(np.count_nonzero(self.grid))
        total = self.size ** 3

        print("World summary")
        print("-------------")
        print(f"World size: {self.size} x {self.size} x {self.size}")
        print(f"Occupied blocks: {occupied}")
        print(f"Occupied ratio: {occupied / total * 100:.2f}%")


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def neighbors_6(p):
    x, y, z = p

    return [
        (x + 1, y, z),
        (x - 1, y, z),
        (x, y + 1, z),
        (x, y - 1, z),
        (x, y, z + 1),
        (x, y, z - 1),
    ]


def astar(world, start, goal):
    open_heap = []
    heapq.heappush(open_heap, (0, start))

    came_from = {}
    g_score = {start: 0}

    visited = set()
    explored_order = []

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current in visited:
            continue

        visited.add(current)
        explored_order.append(current)

        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, explored_order

        for nb in neighbors_6(current):
            if not world.inside(*nb):
                continue

            if world.is_occupied(nb):
                continue

            tentative_g = g_score[current] + 1

            if tentative_g < g_score.get(nb, float("inf")):
                came_from[nb] = current
                g_score[nb] = tentative_g
                f_score = tentative_g + heuristic(nb, goal)
                heapq.heappush(open_heap, (f_score, nb))

    return None, explored_order


def reconstruct_path(came_from, current):
    path = [current]

    while current in came_from:
        current = came_from[current]
        path.append(current)

    path.reverse()
    return path


def create_kalman_filter(start):
    F = np.array([
        [1.0, 0.0, 0.0, DT,  0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, DT,  0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, DT ],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ])

    G = np.zeros((6, 3))
    H = np.eye(6)

    P = np.eye(6) * 1.0

    Q = np.diag([
        0.05, 0.05, 0.05,
        0.15, 0.15, 0.15,
    ])

    R = np.diag([
        SENSOR_SIGMA_POS ** 2,
        SENSOR_SIGMA_POS ** 2,
        SENSOR_SIGMA_POS ** 2,
        SENSOR_SIGMA_VEL ** 2,
        SENSOR_SIGMA_VEL ** 2,
        SENSOR_SIGMA_VEL ** 2,
    ])

    X = np.array([
        float(start[0]),
        float(start[1]),
        float(start[2]),
        0.0,
        0.0,
        0.0,
    ])

    return Kalman(F, G, H, P, Q, R, X)


class DroneAgent:
    def __init__(self, world, start, goal):
        self.world = world
        self.start = start
        self.goal = goal

        self.real_position = np.array(start, dtype=float)
        self.previous_real_position = self.real_position.copy()

        self.estimated_position = np.array(start, dtype=float)
        self.measurement_position = np.array(start, dtype=float)

        self.kf = create_kalman_filter(start)

        self.path = None
        self.explored_nodes = []

        self.current_path_index = 0
        self.path_length_blocks = 0

        self.true_history = []
        self.estimate_history = []
        self.measurement_history = []

        self.travelled_distance = 0.0
        self.last_position_for_distance = self.real_position.copy()

        self.finished = False
        self.collision = False

    def distance_to_goal(self):
        return float(np.linalg.norm(self.real_position - np.array(self.goal, dtype=float)))

    def has_path(self):
        return self.path is not None and len(self.path) > 0

    def plan_path(self):
        print("Planning path with A*...")

        path, explored = astar(self.world, self.start, self.goal)
        self.explored_nodes = explored

        if path is None:
            print("A* failed: no path found.")
            return False

        self.path = [np.array(p, dtype=float) for p in path]
        self.current_path_index = 0
        self.path_length_blocks = len(self.path)

        print("A* path found.")
        print(f"Path length: {self.path_length_blocks} blocks")
        print(f"Explored nodes: {len(self.explored_nodes)}")

        return True

    def validate_path(self):
        if self.path is None:
            return

        for i in range(len(self.path) - 1):
            a = self.path[i]
            b = self.path[i + 1]
            mid = tuple(np.round((a + b) / 2).astype(int))

            if self.world.is_occupied(mid):
                print(f"Path collision between waypoint {i} and {i+1}: mid={mid}")

    def target_waypoint(self):
        if not self.has_path():
            return None

        if self.current_path_index >= len(self.path):
            return None

        return self.path[self.current_path_index]

    def move_towards_waypoint(self):
        target = self.target_waypoint()

        if target is None:
            return

        direction = target - self.estimated_position
        distance = np.linalg.norm(direction)

        if distance < 0.2:
            self.current_path_index += 1
            return

        velocity = direction / distance * MAX_SPEED

        old_position = self.real_position.copy()
        new_position = self.real_position + velocity * DT

        new_position[0] = np.clip(new_position[0], 0, self.world.size - 1)
        new_position[1] = np.clip(new_position[1], 0, self.world.size - 1)
        new_position[2] = np.clip(new_position[2], 0, self.world.size - 1)

        rounded = tuple(np.round(new_position).astype(int))

        if self.world.is_occupied(rounded):
            print("\nCollision risk detected. Stopping.")
            self.finished = True
            self.collision = True
            return

        self.real_position = new_position

        step_distance = float(np.linalg.norm(self.real_position - old_position))
        self.travelled_distance += step_distance

    def sensor_measurement(self):
        velocity = (self.real_position - self.previous_real_position) / DT
        self.previous_real_position = self.real_position.copy()

        z = np.array([
            self.real_position[0] + random.gauss(0, SENSOR_SIGMA_POS),
            self.real_position[1] + random.gauss(0, SENSOR_SIGMA_POS),
            self.real_position[2] + random.gauss(0, SENSOR_SIGMA_POS),
            velocity[0] + random.gauss(0, SENSOR_SIGMA_VEL),
            velocity[1] + random.gauss(0, SENSOR_SIGMA_VEL),
            velocity[2] + random.gauss(0, SENSOR_SIGMA_VEL),
        ])

        self.measurement_position = z[:3]
        return z

    def kalman_update(self):
        z = self.sensor_measurement()
        u = np.array([0.0, 0.0, 0.0])

        predicted = self.kf.predict(u)
        self.kf.update(predicted, z)

        self.estimated_position = np.array([
            float(self.kf.X[0]),
            float(self.kf.X[1]),
            float(self.kf.X[2]),
        ])

    def record_history(self):
        self.true_history.append(self.real_position.copy())
        self.estimate_history.append(self.estimated_position.copy())
        self.measurement_history.append(self.measurement_position.copy())

    def remaining_waypoints(self):
        if self.path is None:
            return 0
        return max(0, len(self.path) - self.current_path_index)

    def progress_ratio(self):
        if self.path is None or len(self.path) == 0:
            return 0.0
        return min(1.0, self.current_path_index / len(self.path))

    def distance_to_current_waypoint(self):
        target = self.target_waypoint()
        if target is None:
            return 0.0
        return float(np.linalg.norm(self.real_position - target))

    def tick(self):
        root = Selector([
            IsGoalReached(self),
            Sequence([
                EnsurePath(self),
                UpdateKalman(self),
                FollowPath(self),
                RecordHistory(self),
            ]),
        ])

        return root.tick()


class BehaviourNode:
    def tick(self):
        raise NotImplementedError


class Sequence(BehaviourNode):
    def __init__(self, children):
        self.children = children

    def tick(self):
        for child in self.children:
            status = child.tick()

            if status != Status.SUCCESS:
                return status

        return Status.SUCCESS


class Selector(BehaviourNode):
    def __init__(self, children):
        self.children = children

    def tick(self):
        for child in self.children:
            status = child.tick()

            if status == Status.SUCCESS:
                return Status.SUCCESS

            if status == Status.RUNNING:
                return Status.RUNNING

        return Status.FAILURE


class IsGoalReached(BehaviourNode):
    def __init__(self, drone):
        self.drone = drone

    def tick(self):
        if self.drone.finished:
            return Status.SUCCESS

        if self.drone.distance_to_goal() < 1.0:
            print("\nGoal reached.")
            self.drone.finished = True
            return Status.SUCCESS

        return Status.FAILURE


class EnsurePath(BehaviourNode):
    def __init__(self, drone):
        self.drone = drone

    def tick(self):
        if self.drone.has_path():
            return Status.SUCCESS

        if self.drone.plan_path():
            return Status.SUCCESS

        self.drone.finished = True
        return Status.FAILURE


class UpdateKalman(BehaviourNode):
    def __init__(self, drone):
        self.drone = drone

    def tick(self):
        self.drone.kalman_update()
        return Status.SUCCESS


class FollowPath(BehaviourNode):
    def __init__(self, drone):
        self.drone = drone

    def tick(self):
        self.drone.move_towards_waypoint()
        return Status.SUCCESS


class RecordHistory(BehaviourNode):
    def __init__(self, drone):
        self.drone = drone

    def tick(self):
        self.drone.record_history()
        return Status.SUCCESS


def create_cube_mesh(block_positions):
    vertices = []
    faces = []
    colors = []

    cube_vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ], dtype=float)

    cube_faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [1, 2, 6], [1, 6, 5],
        [0, 3, 7], [0, 7, 4],
    ])

    base_color = np.array([0.35, 0.40, 0.55, 1.0])

    for block in block_positions:
        start_index = len(vertices)
        cube = cube_vertices + block.astype(float)

        vertices.extend(cube)
        faces.extend(cube_faces + start_index)

        for _ in range(len(cube_faces)):
            colors.append(base_color)

    return (
        np.array(vertices, dtype=float),
        np.array(faces, dtype=np.uint32),
        np.array(colors, dtype=float),
    )


class SimulationViewer:
    def __init__(self, world, drone):
        self.world = world
        self.drone = drone

        self.frame_counter = 0

        self.canvas = scene.SceneCanvas(
            keys="interactive",
            show=True,
            title="A* + Behaviour Tree + Kalman Drone Simulation",
            size=(1200, 900),
            bgcolor="black",
        )

        self.view = self.canvas.central_widget.add_view()

        self.view.camera = scene.cameras.TurntableCamera(
            fov=60,
            azimuth=45,
            elevation=30,
            distance=145,
            center=(
                world.size / 2,
                world.size / 2,
                world.size / 2,
            ),
        )

        self.add_world_mesh()
        self.add_path_line()

        self.true_marker = visuals.Markers(parent=self.view.scene)
        self.estimate_marker = visuals.Markers(parent=self.view.scene)
        self.measurement_marker = visuals.Markers(parent=self.view.scene)
        self.start_goal_marker = visuals.Markers(parent=self.view.scene)
        self.current_waypoint_marker = visuals.Markers(parent=self.view.scene)

        self.true_line = visuals.Line(parent=self.view.scene)
        self.estimate_line = visuals.Line(parent=self.view.scene)
        self.connection_line = visuals.Line(parent=self.view.scene)
        self.heading_line = visuals.Line(parent=self.view.scene)

        visuals.XYZAxis(parent=self.view.scene)

        self.draw_start_goal()

        self.timer = app.Timer(
            interval=DT,
            connect=self.on_timer,
            start=True,
        )

    def add_world_mesh(self):
        occupied = self.world.occupied_coordinates()

        if len(occupied) > MAX_RENDER_BLOCKS:
            indices = np.random.choice(
                len(occupied),
                size=MAX_RENDER_BLOCKS,
                replace=False,
            )
            occupied = occupied[indices]

        vertices, faces, colors = create_cube_mesh(occupied)

        visuals.Mesh(
            vertices=vertices,
            faces=faces,
            face_colors=colors,
            shading="flat",
            parent=self.view.scene,
        )

    def add_explored_nodes(self):
        if not self.drone.explored_nodes:
            return

        explored = np.array(self.drone.explored_nodes, dtype=float)

        max_explored_render = 7000

        if len(explored) > max_explored_render:
            indices = np.random.choice(
                len(explored),
                size=max_explored_render,
                replace=False,
            )
            explored = explored[indices]

        explored_markers = visuals.Markers(parent=self.view.scene)
        explored_markers.set_data(
            explored,
            face_color=(1.0, 0.35, 0.0, 0.35),
            size=3,
        )

    def add_path_line(self):
        if self.drone.path is None:
            return

        path_points = np.array(self.drone.path, dtype=float)

        visuals.Line(
            pos=path_points,
            color=(0.0, 0.9, 1.0, 0.2),
            width=4,
            parent=self.view.scene,
        )

    def draw_start_goal(self):
        points = np.array([
            self.drone.start,
            self.drone.goal,
        ], dtype=float)

        colors = np.array([
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
        ])

        self.start_goal_marker.set_data(
            points,
            face_color=colors,
            size=18,
        )

    def on_timer(self, event):
        import traceback

        try:
            self._do_update()
        except Exception:
            traceback.print_exc()
            self.timer.stop()

    def _do_update(self):
        self.frame_counter += 1

        # Run multiple simulation steps per frame for faster progress
        steps_per_frame = 5
        for _ in range(steps_per_frame):
            if not self.drone.finished:
                self.drone.tick()

        pos = self.drone.real_position.copy()
        est = self.drone.estimated_position.copy()
        meas = self.drone.measurement_position.copy()

        # Real drone
        self.true_marker.set_data(
            np.array([pos]),
            face_color=(0.0, 1.0, 0.0, 1.0),
            size=22,
        )

        # Kalman estimate
        self.estimate_marker.set_data(
            np.array([est]),
            face_color=(0.0, 0.2, 1.0, 1.0),
            size=16,
        )

        # Noisy measurement
        self.measurement_marker.set_data(
            np.array([meas]),
            face_color=(1.0, 0.0, 0.0, 0.8),
            size=10,
        )

        # Current waypoint
        target = self.drone.target_waypoint()

        if target is not None:
            self.current_waypoint_marker.set_data(
                np.array([target]),
                face_color=(1.0, 0.0, 1.0, 1.0),
                size=16,
            )

            self.connection_line.set_data(
                np.array([pos, target]),
                color=(1.0, 0.0, 1.0, 1.0),
                width=2,
            )

        # Heading line
        if len(self.drone.true_history) > 2:
            direction = self.drone.real_position - self.drone.true_history[-2]
            norm = np.linalg.norm(direction)

            if norm > 1e-5:
                direction = direction / norm
                heading_end = pos + direction * 5.0

                self.heading_line.set_data(
                    np.array([pos, heading_end]),
                    color=(0.0, 1.0, 0.0, 1.0),
                    width=4,
                )

        # True trajectory
        if len(self.drone.true_history) > 1:
            self.true_line.set_data(
                np.array(self.drone.true_history),
                color=(0.0, 1.0, 0.0, 0.8),
                width=3,
            )

        # Estimated trajectory
        if len(self.drone.estimate_history) > 1:
            self.estimate_line.set_data(
                np.array(self.drone.estimate_history),
                color=(0.0, 0.2, 1.0, 0.7),
                width=2,
            )

        if self.frame_counter % 10 == 0:
            print(
                f"\r"
                f"Progress: {self.drone.progress_ratio() * 100:5.1f}% | "
                f"Travelled: {self.drone.travelled_distance:7.2f} m | "
                f"Remaining waypoints: {self.drone.remaining_waypoints():4d} | "
                f"Distance to waypoint: {self.drone.distance_to_current_waypoint():6.2f} m | "
                f"A* path blocks: {self.drone.path_length_blocks:4d} | "
                f"A* explored: {len(self.drone.explored_nodes):5d} | "
                f"Drone pos: ({pos[0]:5.1f}, {pos[1]:5.1f}, {pos[2]:5.1f})",
                end="",
                flush=True,
            )

        self.canvas.update()


def generate_valid_world():
    attempt = 0

    while True:
        attempt += 1

        world = BlockWorld(WORLD_SIZE)
        world.generate_random_obstacles(NUM_OBSTACLES)

        start = world.random_free_cell()
        goal = world.random_free_cell()

        if heuristic(start, goal) < 150:
            continue

        print(f"Attempt {attempt}: start={start}, goal={goal}")

        path, explored = astar(world, start, goal)

        if path is not None:
            print("Valid world found.")
            print(f"A* path length: {len(path)} blocks")
            print(f"A* explored nodes: {len(explored)}")
            return world, start, goal

        print("No path in this world. Regenerating...")


def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    world, start, goal = generate_valid_world()
    world.print_summary()

    drone = DroneAgent(world, start, goal)
    drone.plan_path()

    print("\nLegend:")
    print("gray blocks   = obstacles (semi-transparent)")
    print("cyan line     = A* path")
    print("green dot     = real drone")
    print("blue dot      = Kalman estimate")
    print("red dot       = noisy sensor measurement")
    print("yellow dot    = goal")
    print("magenta dot   = current waypoint")
    print("magenta line  = drone -> current waypoint")
    print("\nControls:")
    print("- mouse drag: rotate")
    print("- mouse wheel: zoom")
    print("- right drag: pan")
    print("\nSimulation starting...\n")

    viewer = SimulationViewer(world, drone)

    app.run()


if __name__ == "__main__":
    main()