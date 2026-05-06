import random
import numpy as np

from vispy import app, scene
from vispy.scene import visuals


WORLD_SIZE = 100
NUM_OBSTACLES = 25
MAX_RENDER_BLOCKS = 20000


class BlockWorld:
    def __init__(self, size=100):
        self.size = size
        self.grid = np.zeros((size, size, size), dtype=bool)

    def is_inside_world(self, x, y, z):
        return 0 <= x < self.size and 0 <= y < self.size and 0 <= z < self.size

    def add_block(self, x, y, z):
        if self.is_inside_world(x, y, z):
            self.grid[x, y, z] = True

    def add_box(self, x, y, z, width, depth, height):
        x2 = min(x + width, self.size)
        y2 = min(y + depth, self.size)
        z2 = min(z + height, self.size)
        self.grid[x:x2, y:y2, z:z2] = True

    def add_sphere_blob(self, cx, cy, cz, radius):
        r2 = radius * radius

        for x in range(cx - radius, cx + radius + 1):
            for y in range(cy - radius, cy + radius + 1):
                for z in range(cz - radius, cz + radius + 1):
                    if not self.is_inside_world(x, y, z):
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

    def generate_random_obstacles(self, count=25, seed=42):
        random.seed(seed)

        for _ in range(count):
            obstacle_type = random.choice([
                "box",
                "sphere_blob",
                "l_shape",
                "tower",
            ])

            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            z = random.randint(0, self.size - 1)

            if obstacle_type == "box":
                self.add_box(
                    x, y, z,
                    width=random.randint(4, 16),
                    depth=random.randint(4, 16),
                    height=random.randint(4, 20),
                )

            elif obstacle_type == "sphere_blob":
                self.add_sphere_blob(
                    x, y, z,
                    radius=random.randint(3, 10),
                )

            elif obstacle_type == "l_shape":
                self.add_l_shape(
                    x, y, z,
                    length=random.randint(8, 22),
                    thickness=random.randint(2, 5),
                    height=random.randint(4, 18),
                )

            elif obstacle_type == "tower":
                self.add_tower(
                    x, y, z,
                    width=random.randint(3, 8),
                    height=random.randint(15, 45),
                )

    def occupied_coordinates(self):
        return np.argwhere(self.grid)

    def print_summary(self):
        occupied = np.count_nonzero(self.grid)
        total = self.size ** 3

        print("Block World Summary")
        print("-------------------")
        print(f"World size: {self.size} x {self.size} x {self.size}")
        print(f"Total blocks: {total}")
        print(f"Occupied blocks: {occupied}")
        print(f"Occupied ratio: {occupied / total * 100:.2f}%")


def create_cube_mesh(block_positions, block_size=1.0):
    vertices = []
    faces = []
    face_colors = []

    # Cube vertices relative to one block position
    cube_vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ], dtype=float) * block_size

    cube_faces = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [2, 3, 7], [2, 7, 6],  # back
        [1, 2, 6], [1, 6, 5],  # right
        [0, 3, 7], [0, 7, 4],  # left
    ])

    base_color = np.array([0.55, 0.55, 0.60, 1.0])

    for block in block_positions:
        start_index = len(vertices)

        cube = cube_vertices + block.astype(float)
        vertices.extend(cube)
        faces.extend(cube_faces + start_index)

        for _ in range(len(cube_faces)):
            face_colors.append(base_color)

    return (
        np.array(vertices, dtype=float),
        np.array(faces, dtype=np.uint32),
        np.array(face_colors, dtype=float),
    )


def render_vispy(world):
    occupied = world.occupied_coordinates()

    if len(occupied) > MAX_RENDER_BLOCKS:
        indices = np.random.choice(
            len(occupied),
            size=MAX_RENDER_BLOCKS,
            replace=False,
        )
        occupied = occupied[indices]

    vertices, faces, colors = create_cube_mesh(occupied)

    canvas = scene.SceneCanvas(
        keys="interactive",
        show=True,
        title="3D Block World - VisPy",
        size=(1200, 900),
    )

    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(
        fov=60,
        azimuth=45,
        elevation=30,
        distance=180,
        center=(world.size / 2, world.size / 2, world.size / 2),
    )

    mesh = visuals.Mesh(
        vertices=vertices,
        faces=faces,
        face_colors=colors,
        shading="flat",
    )

    view.add(mesh)

    axis = visuals.XYZAxis(parent=view.scene)

    print("Controls:")
    print("- Mouse drag: rotate")
    print("- Mouse wheel: zoom")
    print("- Right mouse drag: pan")

    app.run()


def main():
    world = BlockWorld(size=WORLD_SIZE)
    world.generate_random_obstacles(
        count=NUM_OBSTACLES,
        seed=42,
    )

    world.print_summary()
    render_vispy(world)


if __name__ == "__main__":
    main()