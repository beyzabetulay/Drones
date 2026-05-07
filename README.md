# 🚁 Drones - 3D Autonomous Drone Simulation

This project is an advanced simulation of an autonomous drone navigating to a target within a complex 3D "Block World". The drone finds its way through an obstacle-filled environment using the **A* (A-Star)** algorithm, makes logical decisions using **Behaviour Trees**, and employs a **Kalman Filter** to filter noisy sensor data. The simulation's visualization is powered by the high-performance **VisPy** library.

---

## ✨ Features

- **Advanced Path Planning (A*):** The drone dynamically calculates the shortest and safest path to reach the goal among complex 3D obstacles using the A* algorithm.
- **Autonomous Decision Making (Behaviour Trees):** The drone's workflow (path planning, Kalman updates, path following, goal checking) is managed by a Behaviour Tree architecture, commonly used in modern gaming and robotics systems.
- **Realistic Sensor Modeling & Kalman Filter:** The drone's sensors (position and velocity) are subjected to random Gaussian noise. This real-world uncertainty is minimized using a Kalman Filter, estimating the drone's true position with high accuracy.
- **Dynamic 3D World Generation:** In each simulation, a random 100x100x100 3D block world is generated, consisting of various obstacles like boxes, L-shapes, towers, and spheres.
- **High-Performance Visualization:** The 3D scene is rendered smoothly using VisPy. The true position, estimated position, and noisy sensor data can be monitored on the screen in real-time.

---

## 🛠️ Technologies and Algorithms

- **Python 3.x**
- **NumPy:** For vector calculations, matrix operations, and Kalman matrices.
- **VisPy:** For GPU-accelerated, real-time 3D visualization.
- **Matplotlib:** (Optional) For basic block world visualizations.
- **A* Algorithm:** Heuristic search in 3D space (3D Grid).
- **Behaviour Tree (BT):** Action control of the drone using Sequence and Selector nodes.
- **Kalman Filter:** State estimation for linear dynamic systems.

---

## 📂 Project Structure

- `drone_world_sim.py`: **The main simulation file.** It generates the world, finds the path using A*, runs the Kalman filter, and renders everything with VisPy.
- `kalman.py`: A pure NumPy implementation of the classic Kalman Filter (Predict and Update phases).
- `block_world.py`: Contains classes to generate a 3D grid world filled with random obstacles. It also includes a Matplotlib-based basic visualization function.
- `block_world_vispy.py`: An alternative structure for visualizing the block world on VisPy.

---

## 🚀 Installation and Usage

### 1. Virtual Environment Setup
It is recommended to use a virtual environment to isolate project dependencies (A `.venv` already exists in the project).

```bash
# Activate the virtual environment
source .venv/bin/activate.fish  # For Fish shell
# or
source .venv/bin/activate       # For Bash/Zsh
```

### 2. Installing Dependencies
If dependencies are not installed, you can easily install them using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
*(Note: VisPy may require PyQt6, PyQt5, or a similar Qt backend to be installed on your system. PyQt6 is included in the requirements).*

### 3. Starting the Simulation
To run the main simulation:
```bash
python drone_world_sim.py
```
When the simulation starts, the A* algorithm's solution details and real-time progress of the simulation (Progress, Travelled, Drone pos, etc.) will be displayed in the terminal.

---

## 🎮 Controls (VisPy Window)

- **Left Click + Drag:** Rotate the camera.
- **Mouse Wheel (Scroll):** Zoom in/out.
- **Right Click + Drag:** Pan the camera.

---

## 📖 Legend (What You See on Screen)

Meanings of the colors in the simulation window:

- ⬛ **Semi-transparent Gray Blocks:** 3D obstacles to avoid.
- 🟦 **Cyan Thick Line:** The ideal path calculated by the A* algorithm.
- 🟢 **Green Dot:** The drone's true position.
- 🔴 **Red Dot:** Noisy and inaccurate measurement from the sensor.
- 🔵 **Blue Dot:** The **estimated position** obtained by filtering the noise via the Kalman Filter.
- 🟡 **Yellow Dot:** The final destination/goal.
- 🟣 **Magenta Dot:** The next waypoint the drone is currently heading towards.
- 🟣 **Magenta Line:** The directional line between the drone and the next waypoint.

---

## ⚙️ Customization

You can test the simulation under different scenarios by changing the constants (parameters) at the top of the `drone_world_sim.py` file:

- `WORLD_SIZE`: The size of the world (default: 100).
- `NUM_OBSTACLES`: The number of obstacles. Determines how crowded the world will be.
- `DT`: Simulation time step (Delta Time).
- `MAX_SPEED`: The speed of the drone.
- `SENSOR_SIGMA_POS` / `SENSOR_SIGMA_VEL`: The amount of random noise in the sensor (deviation/standard deviation value). You can increase these values to test how well the Kalman filter performs.
- **A* Clearance Modification:** To prevent collisions, the safe clearance distance around the drone can be adjusted via the "clearance" parameter in the `is_safe_cell` function within the code.

---

## 👨‍💻 Development Notes
- This system successfully brings together the three main pillars of robotics systems (**Perception/Estimation, Planning, Control**) under a single roof.
- The drone's movement is kinematically modeled, and animations are updated asynchronously via timer callbacks.

