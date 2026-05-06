import random
import numpy as np
import matplotlib.pyplot as plt


WORLD_SIZE = 50
NUM_OBSTACLES = 25
MAX_RENDER_BLOCKS = 8000


class BlockWorld:
    def __init__(self, size=100):
        self.size = size

        # False = empty block
        # True  = occupied block / obstacle
        self.grid = np.zeros((size, size, size), dtype=bool)

    def is_inside_world(self, x, y, z):
        return (
            0 <= x < self.size
            and 0 <= y < self.size
            and 0 <= z < self.size
        )

    def add_block(self, x, y, z):
        if self.is_inside_world(x, y, z):
            self.grid[x, y, z] = True

    def add_box(self, x, y, z, width, depth, height):
        """
        Rectangular cuboid obstacle.
        """
        for i in range(x, x + width):
            for j in range(y, y + depth):
                for k in range(z, z + height):
                    self.add_block(i, j, k)

    def add_sphere_like_blob(self, cx, cy, cz, radius):
        """
        Blocky sphere-like obstacle.
        Still made from discrete blocks.
        """
        r2 = radius * radius

        for x in range(cx - radius, cx + radius + 1):
            for y in range(cy - radius, cy + radius + 1):
                for z in range(cz - radius, cz + radius + 1):
                    dx = x - cx
                    dy = y - cy
                    dz = z - cz

                    if dx * dx + dy * dy + dz * dz <= r2:
                        self.add_block(x, y, z)

    def add_l_shape(self, x, y, z, length, thickness, height):
        """
        Blocky L-shaped obstacle.
        """
        self.add_box(x, y, z, length, thickness, height)
        self.add_box(x, y, z, thickness, length, height)

    def add_tower(self, x, y, z, width, height):
        """
        Vertical block tower.
        """
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
                width = random.randint(4, 16)
                depth = random.randint(4, 16)
                height = random.randint(4, 20)
                self.add_box(x, y, z, width, depth, height)

            elif obstacle_type == "sphere_blob":
                radius = random.randint(3, 10)
                self.add_sphere_like_blob(x, y, z, radius)

            elif obstacle_type == "l_shape":
                length = random.randint(8, 22)
                thickness = random.randint(2, 5)
                height = random.randint(4, 18)
                self.add_l_shape(x, y, z, length, thickness, height)

            elif obstacle_type == "tower":
                width = random.randint(3, 8)
                height = random.randint(15, 45)
                self.add_tower(x, y, z, width, height)

    def get_occupied_coordinates(self):
        return np.argwhere(self.grid)

    def print_summary(self):
        occupied = np.count_nonzero(self.grid)
        total = self.size ** 3
        ratio = occupied / total * 100

        print("Block World Summary")
        print("-------------------")
        print(f"World size: {self.size} x {self.size} x {self.size}")
        print(f"Total blocks: {total}")
        print(f"Occupied blocks: {occupied}")
        print(f"Occupied ratio: {ratio:.2f}%")


def render_world(world):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.voxels(
        world.grid,
        edgecolor="k",
        linewidth=0.1,
        alpha=0.7,
    )

    ax.set_xlim(0, world.size)
    ax.set_ylim(0, world.size)
    ax.set_zlim(0, world.size)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title("100 x 100 x 100 Block World with Block Obstacles")

    plt.show()


def main():
    world = BlockWorld(size=WORLD_SIZE)
    world.generate_random_obstacles(
        count=NUM_OBSTACLES,
        seed=42,
    )

    world.print_summary()
    render_world(world)


if __name__ == "__main__":
    main()