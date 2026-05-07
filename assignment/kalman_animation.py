import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from kalman import Kalman


DT = 0.1          # 10 Hz
MAX_SPEED = 1.0
R_VAR_POS = 0.20
R_VAR_VEL = 0.10


def create_kalman_filter(start):
    F = np.array([
        [1.0, 0.0, DT,  0.0],
        [0.0, 1.0, 0.0, DT ],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    G = np.zeros((4, 2))
    H = np.eye(4)

    P = np.eye(4)
    Q = np.diag([0.01, 0.01, 0.05, 0.05])
    R = np.diag([R_VAR_POS, R_VAR_POS, R_VAR_VEL, R_VAR_VEL])

    X = np.array([start[0], start[1], 0.0, 0.0])

    return Kalman(F, G, H, P, Q, R, X)


def limit_speed(vx, vy):
    speed = np.sqrt(vx**2 + vy**2)
    if speed > MAX_SPEED:
        vx = vx / speed * MAX_SPEED
        vy = vy / speed * MAX_SPEED
    return vx, vy


def controller(position, target):
    dx = target[0] - position[0]
    dy = target[1] - position[1]

    distance = np.sqrt(dx**2 + dy**2)

    if distance < 0.05:
        return 0.0, 0.0

    vx = dx / distance * MAX_SPEED
    vy = dy / distance * MAX_SPEED

    return limit_speed(vx, vy)


def simulate_sensor(true_state, rng):
    noise = np.array([
        rng.normal(0, np.sqrt(R_VAR_POS)),
        rng.normal(0, np.sqrt(R_VAR_POS)),
        rng.normal(0, np.sqrt(R_VAR_VEL)),
        rng.normal(0, np.sqrt(R_VAR_VEL)),
    ])

    return true_state + noise


# L-shape
waypoints = np.array([
    [1.0, 1.0],
    [1.0, 8.0],
    [8.0, 8.0],
])

# U-shape istersen bunu aç:
# waypoints = np.array([
#     [1.0, 8.0],
#     [1.0, 1.0],
#     [8.0, 1.0],
#     [8.0, 8.0],
# ])

rng = np.random.default_rng(42)

true_state = np.array([waypoints[0, 0], waypoints[0, 1], 0.0, 0.0])
kf = create_kalman_filter(waypoints[0])

current_target_index = 1

true_history = []
measurement_history = []
estimate_history = []

fig, ax = plt.subplots(figsize=(8, 8))

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.grid(True)

ax.set_title("Kalman Filter Drone Simulation")
ax.set_xlabel("x position [m]")
ax.set_ylabel("y position [m]")

# reference path
ax.plot(
    waypoints[:, 0],
    waypoints[:, 1],
    "o--",
    label="Reference path"
)

true_dot, = ax.plot([], [], "o", markersize=10, label="True position")
measurement_dot, = ax.plot([], [], "s", markersize=6, alpha=0.6, label="Noisy sensor")
estimate_dot, = ax.plot([], [], "o", markersize=8, label="Kalman estimate")

true_line, = ax.plot([], [], linewidth=2, label="True trajectory")
estimate_line, = ax.plot([], [], linewidth=2, label="Estimated trajectory")

target_dot, = ax.plot([], [], "X", markersize=12, label="Current target")

ax.legend(loc="upper left")


def update(frame):
    global true_state, current_target_index

    if current_target_index < len(waypoints):
        target = waypoints[current_target_index]

        # controller uses Kalman estimate
        estimated_position = kf.X[:2]
        vx, vy = controller(estimated_position, target)

        true_state[2] = vx
        true_state[3] = vy

        true_state[0] += true_state[2] * DT
        true_state[1] += true_state[3] * DT

        true_state[0] = np.clip(true_state[0], 0.0, 10.0)
        true_state[1] = np.clip(true_state[1], 0.0, 10.0)

        Z = simulate_sensor(true_state, rng)

        U = np.array([0.0, 0.0])
        predicted_X = kf.predict(U)
        kf.update(predicted_X, Z)

        if np.linalg.norm(true_state[:2] - target) < 0.25:
            current_target_index += 1
    else:
        Z = simulate_sensor(true_state, rng)
        U = np.array([0.0, 0.0])
        predicted_X = kf.predict(U)
        kf.update(predicted_X, Z)

    true_history.append(true_state[:2].copy())
    measurement_history.append(Z[:2].copy())
    estimate_history.append(kf.X[:2].copy())

    true_arr = np.array(true_history)
    meas_arr = np.array(measurement_history)
    est_arr = np.array(estimate_history)

    true_dot.set_data([true_state[0]], [true_state[1]])
    measurement_dot.set_data([Z[0]], [Z[1]])
    estimate_dot.set_data([kf.X[0]], [kf.X[1]])

    true_line.set_data(true_arr[:, 0], true_arr[:, 1])
    estimate_line.set_data(est_arr[:, 0], est_arr[:, 1])

    if current_target_index < len(waypoints):
        target = waypoints[current_target_index]
    else:
        target = waypoints[-1]

    target_dot.set_data([target[0]], [target[1]])

    return (
        true_dot,
        measurement_dot,
        estimate_dot,
        true_line,
        estimate_line,
        target_dot,
    )


animation = FuncAnimation(
    fig,
    update,
    frames=500,
    interval=100,
    blit=True,
    repeat=False,
)

plt.show()