import numpy as np
import matplotlib.pyplot as plt

from kalman import Kalman


DT = 0.1          # 10 Hz
MAX_SPEED = 1.0  # m/s
STEPS = 900

# Noise variances
Q_VAR_POS = 0.01
Q_VAR_VEL = 0.05

R_VAR_POS = 0.20
R_VAR_VEL = 0.10


def limit_speed(vx, vy, max_speed=MAX_SPEED):
    speed = np.sqrt(vx**2 + vy**2)

    if speed > max_speed:
        vx = vx / speed * max_speed
        vy = vy / speed * max_speed

    return vx, vy


def controller(position, target):
    """
    Simple closed-loop bang-bang style controller.
    It always tries to move directly towards the current target waypoint.
    """
    dx = target[0] - position[0]
    dy = target[1] - position[1]

    distance = np.sqrt(dx**2 + dy**2)

    if distance < 0.05:
        return 0.0, 0.0

    vx = dx / distance * MAX_SPEED
    vy = dy / distance * MAX_SPEED

    return limit_speed(vx, vy)


def simulate_sensor(true_state, rng):
    """
    Observe x, y position and vx, vy speed vector with Gaussian noise.
    """
    noise = np.array([
        rng.normal(0, np.sqrt(R_VAR_POS)),
        rng.normal(0, np.sqrt(R_VAR_POS)),
        rng.normal(0, np.sqrt(R_VAR_VEL)),
        rng.normal(0, np.sqrt(R_VAR_VEL)),
    ])

    return true_state + noise


def create_kalman_filter():
    F = np.array([
        [1.0, 0.0, DT,  0.0],
        [0.0, 1.0, 0.0, DT ],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    # No acceleration input in this first version
    G = np.zeros((4, 2))

    # Sensor observes x, y, vx, vy directly
    H = np.eye(4)

    P = np.eye(4) * 1.0

    Q = np.diag([
        Q_VAR_POS,
        Q_VAR_POS,
        Q_VAR_VEL,
        Q_VAR_VEL,
    ])

    R = np.diag([
        R_VAR_POS,
        R_VAR_POS,
        R_VAR_VEL,
        R_VAR_VEL,
    ])

    X = np.array([1.0, 1.0, 0.0, 0.0])

    return Kalman(F, G, H, P, Q, R, X)


def run_simulation(path_name, waypoints):
    rng = np.random.default_rng(42)

    kf = create_kalman_filter()
    start = waypoints[0]

    true_state = np.array([start[0], start[1], 0.0, 0.0])
    kf.X = np.array([start[0], start[1], 0.0, 0.0])

    current_target_index = 1

    true_positions = []
    measured_positions = []
    estimated_positions = []
    target_positions = []

    for _ in range(STEPS):
        target = waypoints[current_target_index]

        # Use estimated position for control, not true position.
        # This makes it closed-loop with state estimation.
        estimated_position_for_control = kf.X[:2]

        cmd_vx, cmd_vy = controller(estimated_position_for_control, target)

        # Simulate real robot/drone motion
        true_state[2] = cmd_vx
        true_state[3] = cmd_vy
        true_state[0] += true_state[2] * DT
        true_state[1] += true_state[3] * DT

        # Keep inside 10m x 10m room
        true_state[0] = np.clip(true_state[0], 0.0, 10.0)
        true_state[1] = np.clip(true_state[1], 0.0, 10.0)

        # Sensor measurement with Gaussian noise
        Z = simulate_sensor(true_state, rng)

        # Kalman predict + update
        U = np.array([0.0, 0.0])
        predicted_X = kf.predict(U)
        kf.update(predicted_X, Z)

        # Check if target reached using estimated position
        dist_to_target = np.linalg.norm(kf.X[:2] - target)

        if dist_to_target < 0.25 and current_target_index < len(waypoints) - 1:
            current_target_index += 1

        true_positions.append(true_state[:2].copy())
        measured_positions.append(Z[:2].copy())
        estimated_positions.append(kf.X[:2].copy())
        target_positions.append(target.copy())

    true_positions = np.array(true_positions)
    measured_positions = np.array(measured_positions)
    estimated_positions = np.array(estimated_positions)

    plt.figure(figsize=(8, 8))

    plt.plot(
        true_positions[:, 0],
        true_positions[:, 1],
        label="True position"
    )

    plt.scatter(
        measured_positions[:, 0],
        measured_positions[:, 1],
        s=5,
        alpha=0.25,
        label="Noisy sensor measurements"
    )

    plt.plot(
        estimated_positions[:, 0],
        estimated_positions[:, 1],
        label="Kalman estimated position"
    )

    wp = np.array(waypoints)

    plt.plot(
        wp[:, 0],
        wp[:, 1],
        "o--",
        label=f"{path_name} reference path"
    )

    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xlabel("x position [m]")
    plt.ylabel("y position [m]")
    plt.title(f"Kalman Filter Closed-Loop Simulation: {path_name}")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def main():
    l_shape = np.array([
        [1.0, 1.0],
        [1.0, 8.0],
        [8.0, 8.0],
    ])

    u_shape = np.array([
        [1.0, 8.0],
        [1.0, 1.0],
        [8.0, 1.0],
        [8.0, 8.0],
    ])

    run_simulation("L-shape", l_shape)
    run_simulation("U-shape", u_shape)


if __name__ == "__main__":
    main()