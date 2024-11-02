def main():
    num_regions = int(input("How many regions to create? "))

    regions = []

    for i in range(num_regions):
        num_points = int(input("How many points for this region? "))

        region = []

        for j in range(num_points):
            x = int(input("X: "))
            y = int(input("Y: "))

            region.append((x, y))

        weight = int(input("What is the weight of this region? "))
        region.append(weight)

        regions.append(region)

    print(regions)


def is_point_on_segment(px, py, p1x, p1y, p2x, p2y):
    """Check if point (px, py) is on the line segment between (p1x, p1y) and (p2x, p2y)"""
    if min(p1x, p2x) <= px <= max(p1x, p2x) and min(p1y, p2y) <= py <= max(p1y, p2y):
        # Check if the points are collinear using the cross-product method
        return (p2y - p1y) * (px - p1x) == (py - p1y) * (p2x - p1x)
    return False


def is_in_region(point, vertices):
    # point (x, y)
    # vertices [(x1, y1), (x2, y2),..., (xn, yn), weight]
    vertices = vertices[:-1]
    print(vertices)
    print(point)

    x, y = point
    n = len(vertices)
    inside = False

    p1x, p1y = vertices[0]
    for i in range(n + 1):
        p2x, p2y = vertices[i % n]

        # Check if the point is exactly on this edge
        if is_point_on_segment(x, y, p1x, p1y, p2x, p2y):
            return True

        # Check if it is inside the boundary
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


if __name__ == "__main__":
    # main()
    region = [(0, 0), (1, 0), (0, 1), 4]
    print(is_in_region((0.51, 0.49), region))
