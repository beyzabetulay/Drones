import heapq
import random
from enum import Enum
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from scipy.interpolate import splprep, splev
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

TRAINING_STEPS = 500_000


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
                self.add_box(x, y, z,
                    random.randint(3, 12),
                    random.randint(3, 12),
                    random.randint(3, 14))
            elif obstacle_type == "sphere":
                self.add_sphere_blob(x, y, z, random.randint(3, 8))
            elif obstacle_type == "l_shape":
                self.add_l_shape(x, y, z,
                    random.randint(6, 16),
                    random.randint(2, 4),
                    random.randint(3, 12))
            elif obstacle_type == "tower":
                self.add_tower(x, y, z,
                    random.randint(3, 6),
                    random.randint(10, 35))

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


def is_safe_cell(world, p, clearance=1):
    x, y, z = p
    for dx in range(-clearance, clearance + 1):
        for dy in range(-clearance, clearance + 1):
            for dz in range(-clearance, clearance + 1):
                if world.is_occupied((x + dx, y + dy, z + dz)):
                    return False
    return True


def neighbors_6(p):
    x, y, z = p
    return [
        (x + 1, y, z), (x - 1, y, z),
        (x, y + 1, z), (x, y - 1, z),
        (x, y, z + 1), (x, y, z - 1),
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
            if not is_safe_cell(world, nb, clearance=1):
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


def smooth_path(path, world=None, smoothing_factor=10):
    if len(path) < 4:
        return path

    points = np.array([p for p in path], dtype=float)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    s = len(points) * smoothing_factor
    tck, u = splprep([x, y, z], s=s, k=3)
    u_fine = np.linspace(0, 1, len(points) * 3)
    smooth_points = splev(u_fine, tck)
    result = np.column_stack(smooth_points)

    if world is None:
        return [result[i] for i in range(len(result))]

    # Obstacle içine giren noktaları orijinal path'e geri döndür
    safe_result = []
    for pt in result:
        rounded = tuple(np.round(pt).astype(int))
        if not is_safe_cell(world, rounded, clearance=1):
            dists = np.linalg.norm(points - pt, axis=1)
            for idx in np.argsort(dists):
                candidate = points[idx]
                c_rounded = tuple(np.round(candidate).astype(int))
                if is_safe_cell(world, c_rounded, clearance=1):
                    safe_result.append(candidate)
                    break
            else:
                safe_result.append(points[0])
        else:
            safe_result.append(pt)

    return safe_result

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
    Q = np.diag([0.05, 0.05, 0.05, 0.15, 0.15, 0.15])
    R = np.diag([
        SENSOR_SIGMA_POS ** 2, SENSOR_SIGMA_POS ** 2, SENSOR_SIGMA_POS ** 2,
        SENSOR_SIGMA_VEL ** 2, SENSOR_SIGMA_VEL ** 2, SENSOR_SIGMA_VEL ** 2,
    ])
    X = np.array([float(start[0]), float(start[1]), float(start[2]), 0.0, 0.0, 0.0])
    return Kalman(F, G, H, P, Q, R, X)


# ─────────────────────────────────────────────
#  Gymnasium environment for RL training
# ─────────────────────────────────────────────

class DroneEnv(gym.Env):
    """
    Drone'un A* path'ini takip etmeyi öğrendiği RL ortamı.

    Observation (15 boyut):
      - normalized estimated position    (3)
      - direction to next waypoint       (3)
      - normalized velocity              (3)
      - obstacle distances in 6 dirs     (6)

    Action (3 boyut, continuous):
      - velocity direction (dx, dy, dz) in [-1, 1]
    """

    metadata = {"render_modes": []}

    def __init__(self, world, start, goal, path):
        super().__init__()

        self.world = world
        self.start = start
        self.goal = goal
        self.path = path  # smooth A* path (list of np.array)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Internal state — filled in reset()
        self.real_position = None
        self.estimated_position = None
        self.previous_position = None
        self.velocity = None
        self.kf = None
        self.current_path_index = 0
        self.steps = 0
        self.prev_dist_to_goal = 0.0
        self.max_steps = 8000

    # ── helpers ──────────────────────────────

    def _dist_to_goal(self):
        return float(np.linalg.norm(
            self.real_position - np.array(self.goal, dtype=float)
        ))

    def _get_obstacle_distances(self):
        """6 yönde obstacle mesafesini ölç, 0-1 arasına normalize et."""
        directions = [
            np.array([1, 0, 0]), np.array([-1, 0, 0]),
            np.array([0, 1, 0]), np.array([0, -1, 0]),
            np.array([0, 0, 1]), np.array([0, 0, -1]),
        ]
        max_range = 10
        distances = []
        for d in directions:
            hit = max_range
            for step in range(1, max_range + 1):
                p = tuple(np.round(self.real_position + d * step).astype(int))
                if self.world.is_occupied(p):
                    hit = step
                    break
            distances.append(hit / max_range)
        return np.array(distances, dtype=np.float32)

    def _get_waypoint_direction(self):
        if self.current_path_index >= len(self.path):
            target = np.array(self.goal, dtype=float)
        else:
            target = np.array(self.path[self.current_path_index], dtype=float)

        direction = target - self.estimated_position
        dist = np.linalg.norm(direction)
        if dist > 1e-6:
            return (direction / dist).astype(np.float32)
        return np.zeros(3, dtype=np.float32)

    def _get_obs(self):
        pos_norm = (self.estimated_position / self.world.size).astype(np.float32)
        waypoint_dir = self._get_waypoint_direction()
        vel_norm = (self.velocity / MAX_SPEED).astype(np.float32)
        obstacle_dists = self._get_obstacle_distances()
        return np.concatenate([pos_norm, waypoint_dir, vel_norm, obstacle_dists])

    def _sensor_measurement(self):
        velocity = (self.real_position - self.previous_position) / DT
        self.previous_position = self.real_position.copy()
        self.velocity = velocity

        z = np.array([
            self.real_position[0] + random.gauss(0, SENSOR_SIGMA_POS),
            self.real_position[1] + random.gauss(0, SENSOR_SIGMA_POS),
            self.real_position[2] + random.gauss(0, SENSOR_SIGMA_POS),
            velocity[0] + random.gauss(0, SENSOR_SIGMA_VEL),
            velocity[1] + random.gauss(0, SENSOR_SIGMA_VEL),
            velocity[2] + random.gauss(0, SENSOR_SIGMA_VEL),
        ])
        return z

    def _kalman_update(self):
        z = self._sensor_measurement()
        u = np.zeros(3)
        predicted = self.kf.predict(u)
        self.kf.update(predicted, z)
        self.estimated_position = np.array([
            float(self.kf.X[0]),
            float(self.kf.X[1]),
            float(self.kf.X[2]),
        ])

    def _advance_waypoint(self):
        if self.current_path_index < len(self.path):
            target = np.array(self.path[self.current_path_index], dtype=float)
            if np.linalg.norm(self.real_position - target) < 1.5:
                self.current_path_index += 1

    # ── gym interface ─────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.real_position = np.array(self.start, dtype=float)
        self.previous_position = self.real_position.copy()
        self.velocity = np.zeros(3)
        self.kf = create_kalman_filter(self.start)
        self.estimated_position = self.real_position.copy()
        self.current_path_index = 0
        self.steps = 0
        self.prev_dist_to_goal = self._dist_to_goal()

        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1

        action = np.array(action, dtype=float)
        norm = np.linalg.norm(action)
        if norm > 1e-6:
            action = action / norm

        velocity = action * MAX_SPEED
        new_position = self.real_position + velocity * DT
        new_position = np.clip(new_position, 0, self.world.size - 1)

        rounded = tuple(np.round(new_position).astype(int))
        collision = bool(self.world.is_occupied(rounded))

        if not collision:
            self.real_position = new_position

        self._kalman_update()

        # Waypoint ilerlemesi
        prev_index = self.current_path_index
        self._advance_waypoint()
        waypoint_advanced = self.current_path_index > prev_index

        # Path'ten sapma mesafesi
        if self.current_path_index < len(self.path):
            target = np.array(self.path[self.current_path_index], dtype=float)
            path_deviation = float(np.linalg.norm(self.real_position - target))
        else:
            path_deviation = 0.0

        # Reward
        curr_dist = self._dist_to_goal()
        progress = self.prev_dist_to_goal - curr_dist
        self.prev_dist_to_goal = curr_dist

        reward = 0.0
        reward += progress * 2.0
        reward += waypoint_advanced * 15.0
        reward -= path_deviation * 0.05
        reward -= 0.1

        if collision:
            reward -= 100.0

        goal_reached = curr_dist < 1.5
        if goal_reached:
            reward += 300.0

        terminated = bool(goal_reached or collision)
        truncated = bool(self.steps >= self.max_steps)

    # Her zaman 5 değer return et
        return self._get_obs(), float(reward), terminated, truncated, {}


# ─────────────────────────────────────────────
#  Drone agent — RL model ile çalışır
# ─────────────────────────────────────────────

class DroneAgent:
    def __init__(self, world, start, goal, rl_model):
        self.world = world
        self.start = start
        self.goal = goal
        self.rl_model = rl_model

        self.real_position = np.array(start, dtype=float)
        self.previous_real_position = self.real_position.copy()

        self.estimated_position = np.array(start, dtype=float)
        self.measurement_position = np.array(start, dtype=float)
        self.velocity = np.zeros(3)

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

    # ── path planning ─────────────────────────

    def plan_path(self):
        print("Planning path with A*...")
        path, explored = astar(self.world, self.start, self.goal)
        self.explored_nodes = explored

        if path is None:
            print("A* failed: no path found.")
            return False

        raw_path = [np.array(p, dtype=float) for p in path]
        self.path = smooth_path(raw_path, self.world, smoothing_factor=10)
        self.current_path_index = 0
        self.path_length_blocks = len(self.path)

        print("A* path found.")
        print(f"Path length (after smoothing): {self.path_length_blocks} waypoints")
        print(f"Explored nodes: {len(self.explored_nodes)}")
        return True

    def has_path(self):
        return self.path is not None and len(self.path) > 0

    def target_waypoint(self):
        if not self.has_path():
            return None
        if self.current_path_index >= len(self.path):
            return None
        return self.path[self.current_path_index]

    # ── observation for RL ────────────────────

    def _get_obs(self):
        pos_norm = (self.estimated_position / self.world.size).astype(np.float32)

        # direction to current waypoint
        if self.current_path_index < len(self.path):
            target = np.array(self.path[self.current_path_index], dtype=float)
        else:
            target = np.array(self.goal, dtype=float)

        direction = target - self.estimated_position
        dist = np.linalg.norm(direction)
        waypoint_dir = (direction / dist).astype(np.float32) if dist > 1e-6 else np.zeros(3, dtype=np.float32)

        vel_norm = (self.velocity / MAX_SPEED).astype(np.float32)

        # obstacle distances in 6 directions
        dirs = [
            np.array([1, 0, 0]), np.array([-1, 0, 0]),
            np.array([0, 1, 0]), np.array([0, -1, 0]),
            np.array([0, 0, 1]), np.array([0, 0, -1]),
        ]
        max_range = 10
        obstacle_dists = []
        for d in dirs:
            hit = max_range
            for step in range(1, max_range + 1):
                p = tuple(np.round(self.real_position + d * step).astype(int))
                if self.world.is_occupied(p):
                    hit = step
                    break
            obstacle_dists.append(hit / max_range)

        return np.concatenate([
            pos_norm, waypoint_dir, vel_norm,
            np.array(obstacle_dists, dtype=np.float32)
        ])

    # ── movement ──────────────────────────────

    def move_with_rl(self):
        obs = self._get_obs()
        action, _ = self.rl_model.predict(obs, deterministic=True)

        action = np.array(action, dtype=float)
        norm = np.linalg.norm(action)
        if norm > 1e-6:
            action = action / norm

        velocity = action * MAX_SPEED
        old_position = self.real_position.copy()
        new_position = self.real_position + velocity * DT

        new_position = np.clip(new_position, 0, self.world.size - 1)
        rounded = tuple(np.round(new_position).astype(int))

        if self.world.is_occupied(rounded):
            return

        self.real_position = new_position
        step_dist = float(np.linalg.norm(self.real_position - old_position))
        self.travelled_distance += step_dist

        # advance waypoint
        if self.current_path_index < len(self.path):
            target = np.array(self.path[self.current_path_index], dtype=float)
            if np.linalg.norm(self.real_position - target) < 1.5:
                self.current_path_index += 1

    def sensor_measurement(self):
        velocity = (self.real_position - self.previous_real_position) / DT
        self.previous_real_position = self.real_position.copy()
        self.velocity = velocity

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
        u = np.zeros(3)
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

    def distance_to_goal(self):
        return float(np.linalg.norm(
            self.real_position - np.array(self.goal, dtype=float)
        ))

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

    # ── behaviour tree tick ───────────────────

    def tick(self):
        root = Selector([
            IsGoalReached(self),
            Sequence([
                EnsurePath(self),
                UpdateKalman(self),
                RLFollowPath(self),
                RecordHistory(self),
            ]),
        ])
        return root.tick()


# ─────────────────────────────────────────────
#  Behaviour tree nodes
# ─────────────────────────────────────────────

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
            print("\nGoal reached!")
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


class RLFollowPath(BehaviourNode):
    """RL modeli ile hareketi yönetir — klasik FollowPath'in yerini alır."""

    def __init__(self, drone):
        self.drone = drone

    def tick(self):
        self.drone.move_with_rl()
        return Status.SUCCESS


class RecordHistory(BehaviourNode):
    def __init__(self, drone):
        self.drone = drone

    def tick(self):
        self.drone.record_history()
        return Status.SUCCESS


# ─────────────────────────────────────────────
#  Visualization
# ─────────────────────────────────────────────

def create_cube_mesh(block_positions):
    vertices = []
    faces = []
    colors = []

    cube_vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
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
            title="A* + Behaviour Tree + Kalman + RL Drone Simulation",
            size=(1200, 900),
            bgcolor="black",
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(
            fov=60, azimuth=45, elevation=30, distance=145,
            center=(world.size / 2, world.size / 2, world.size / 2),
        )

        self._add_world_mesh()
        self._add_path_line()

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
        self._draw_start_goal()

        self.timer = app.Timer(interval=DT, connect=self.on_timer, start=True)

    def _add_world_mesh(self):
        occupied = self.world.occupied_coordinates()
        if len(occupied) > MAX_RENDER_BLOCKS:
            indices = np.random.choice(len(occupied), size=MAX_RENDER_BLOCKS, replace=False)
            occupied = occupied[indices]
        vertices, faces, colors = create_cube_mesh(occupied)
        mesh = visuals.Mesh(
            vertices=vertices, faces=faces,
            face_colors=colors, shading="flat",
            parent=self.view.scene,
        )
        mesh.set_gl_state("opaque", depth_test=True)

    def _add_path_line(self):
        if self.drone.path is None:
            return
        path_points = np.array(self.drone.path, dtype=float)
        visuals.Line(
            pos=path_points,
            color=(0.0, 0.9, 1.0, 0.2),
            width=4,
            parent=self.view.scene,
        )

    def _draw_start_goal(self):
        points = np.array([self.drone.start, self.drone.goal], dtype=float)
        colors = np.array([[0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]])
        self.start_goal_marker.set_data(points, face_color=colors, size=18)

    def on_timer(self, event):
        import traceback
        try:
            self._do_update()
        except Exception:
            traceback.print_exc()
            self.timer.stop()

    def _do_update(self):
        self.frame_counter += 1

        steps_per_frame = 5
        for _ in range(steps_per_frame):
            if not self.drone.finished:
                self.drone.tick()

        pos = self.drone.real_position.copy()
        est = self.drone.estimated_position.copy()
        meas = self.drone.measurement_position.copy()

        self.true_marker.set_data(np.array([pos]), face_color=(0.0, 1.0, 0.0, 1.0), size=22)
        self.estimate_marker.set_data(np.array([est]), face_color=(0.0, 0.2, 1.0, 1.0), size=16)
        self.measurement_marker.set_data(np.array([meas]), face_color=(1.0, 0.0, 0.0, 0.8), size=10)

        target = self.drone.target_waypoint()
        if target is not None:
            self.current_waypoint_marker.set_data(
                np.array([target]), face_color=(1.0, 0.0, 1.0, 1.0), size=16)
            self.connection_line.set_data(
                np.array([pos, target]), color=(1.0, 0.0, 1.0, 1.0), width=2)

        if len(self.drone.true_history) > 2:
            direction = self.drone.real_position - self.drone.true_history[-2]
            norm = np.linalg.norm(direction)
            if norm > 1e-5:
                direction = direction / norm
                self.heading_line.set_data(
                    np.array([pos, pos + direction * 5.0]),
                    color=(0.0, 1.0, 0.0, 1.0), width=4)

        if len(self.drone.true_history) > 1:
            self.true_line.set_data(
                np.array(self.drone.true_history),
                color=(0.0, 1.0, 0.0, 0.8), width=3)

        if len(self.drone.estimate_history) > 1:
            self.estimate_line.set_data(
                np.array(self.drone.estimate_history),
                color=(0.0, 0.2, 1.0, 0.7), width=2)

        if self.frame_counter % 10 == 0:
            print(
                f"\rProgress: {self.drone.progress_ratio() * 100:5.1f}% | "
                f"Travelled: {self.drone.travelled_distance:7.2f} m | "
                f"Remaining waypoints: {self.drone.remaining_waypoints():4d} | "
                f"Dist to waypoint: {self.drone.distance_to_current_waypoint():6.2f} m | "
                f"Drone pos: ({pos[0]:5.1f}, {pos[1]:5.1f}, {pos[2]:5.1f})",
                end="", flush=True,
            )

        self.canvas.update()


# ─────────────────────────────────────────────
#  World generation
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

MODEL_PATH = "drone_rl_model.zip"

def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    # 1) Dünyayı oluştur
    world, start, goal = generate_valid_world()
    world.print_summary()

    # 2) A* ile yolu bul ve smooth et
    print("\nPlanning initial path for training...")
    raw_path, _ = astar(world, start, goal)
    smooth = smooth_path(
        [np.array(p, dtype=float) for p in raw_path],
        world,
        smoothing_factor=10,
    )

    # 3) RL ortamını oluştur
    env = DroneEnv(world, start, goal, smooth)

    # 4) Kayıtlı model varsa yükle, yoksa eğit
    if os.path.exists(MODEL_PATH):
        print(f"\nLoading saved model from {MODEL_PATH}...")
        model = PPO.load(MODEL_PATH, env=env)
        print("Model loaded.")
    else:
        print(f"\nTraining RL agent for {TRAINING_STEPS} steps...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
        )
        model.learn(total_timesteps=TRAINING_STEPS)
        model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}.")

    # 5) Eğitilmiş model ile drone'u başlat
    drone = DroneAgent(world, start, goal, rl_model=model)
    drone.plan_path()

    print("\nLegend:")
    print("gray blocks   = obstacles")
    print("cyan line     = A* smooth path")
    print("green dot     = real drone")
    print("blue dot      = Kalman estimate")
    print("red dot       = noisy sensor measurement")
    print("yellow dot    = goal")
    print("magenta dot   = current waypoint")
    print("magenta line  = drone -> current waypoint")
    print("\nControls: mouse drag=rotate | wheel=zoom | right drag=pan")
    print("\nSimulation starting (RL agent in control)...\n")

    viewer = SimulationViewer(world, drone)
    app.run()


if __name__ == "__main__":
    main()