from perlin_noise import PerlinNoise
import matplotlib.pyplot as plt
import numpy as np
import itertools


def get_connected_cells(array, start):
    # Get the value at the start position
    value = array[start]
    rows, cols = array.shape
    visited = set()  # To keep track of visited cells
    connected_cells = []  # To store the coordinates of connected cells

    def dfs(cell):
        x, y = cell
        # Check if the cell is out of bounds or already visited
        if x < 0 or x >= rows or y < 0 or y >= cols or cell in visited or array[x, y] != value:
            return
        # Mark the cell as visited and add it to the result
        visited.add(cell)
        connected_cells.append((y, x))

        # Explore neighbors (up, down, left, right)
        dfs((x - 1, y))  # Up
        dfs((x + 1, y))  # Down
        dfs((x, y - 1))  # Left
        dfs((x, y + 1))  # Right

    # Start DFS from the starting cell
    dfs(start)
    return connected_cells


def generate_noise_graph(seed):
    noise = PerlinNoise(octaves=5, seed=seed)

    size = 5

    num_layers = 2
    next_max = 1 / num_layers
    layer_thresholds = []
    for i in range(num_layers):
        layer_thresholds.append(round(next_max, 2))
        next_max += 1 / num_layers

    noisemap = np.zeros((size, size))

    for x in range(size):
        for y in range(size):
            noise_value = noise([x / size, y / size]) + 0.5
            if noise_value > 1:
                noise_value = 1
            noisemap[x, y] = noise_value

    for x in range(size):
        for y in range(size):
            noise_value = noisemap[x, y]
            for threshold in layer_thresholds:
                if noise_value <= threshold:
                    noisemap[x, y] = threshold
                    break

    squares_stored = []
    regions = []
    for x, y in itertools.product(range(size), range(size)):
        if (x, y) in squares_stored:
            continue

        noise_value = noisemap[x, y]

        connected = get_connected_cells(noisemap, (y, x))
        squares_stored += connected

        regions.append(connected)

    final_regions = []
    for region in regions:
        region_verts = []
        for square in region:
            x, y = square
            noise_value = noisemap[x, y]
            # check up, right, down, left squares
            if (x, y - 1) not in region:
                region_verts.append((x - 0.5, y - 0.5))
                region_verts.append((x + 0.5, y - 0.5))
            if (x + 1, y) not in region:
                region_verts.append((x + 0.5, y - 0.5))
                region_verts.append((x + 0.5, y + 0.5))
            if (x, y + 1) not in region:
                region_verts.append((x + 0.5, y + 0.5))
                region_verts.append((x - 0.5, y + 0.5))
            if (x - 1, y) not in region:
                region_verts.append((x - 0.5, y + 0.5))
                region_verts.append((x - 0.5, y - 0.5))

        current_pos = region_verts[0]
        actual_verts = [[current_pos[0], current_pos[1]]]
        # try and go up, then right, then down, then left - no doubling back
        while True:
            x, y = current_pos
            if (x, y - 1) in region_verts and [x, y - 1] not in actual_verts:
                actual_verts.append([x, y - 1])
                current_pos = (x, y - 1)
            elif (x + 1, y) in region_verts and [x + 1, y] not in actual_verts:
                actual_verts.append([x + 1, y])
                current_pos = (x + 1, y)
            elif (x, y + 1) in region_verts and [x, y + 1] not in actual_verts:
                actual_verts.append([x, y + 1])
                current_pos = (x, y + 1)
            elif (x - 1, y) in region_verts and [x - 1, y] not in actual_verts:
                actual_verts.append([x - 1, y])
                current_pos = (x - 1, y)
            else:
                break

        for vert in actual_verts:
            vert[1] = -vert[1]

        final_regions.append(
            [(v[0], v[1]) for v in actual_verts] + [float(noise_value * num_layers + 1)]
        )

        return final_regions


if __name__ == "__main__":
    # noisemap = generate_noise_graph(2)
    # plt.imshow(noisemap, cmap="gray")
    # plt.show()
    pass
