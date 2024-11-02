import numpy as np
import math

regions = [[(0, 0), (1, 0), (0, 1), 4], [(1, 0), (0, 1), (1, 1), 3]]
epsilon = (0.00001, 0.00001)


# Get the current region based on a point and vector
# This is to be used when the ray is at an edge
def get_current_region(point, vector):

    # Nudge the position slightly in the direction it's pointing
    point = (point[0] + vector[0] * epsilon[0], point[1] + vector[1] * epsilon[1])
    i = 0
    # Check all the regions, return when the point is in one
    for region in regions:
        if is_in_region(point, region):
            return i
        i += 1

    return None


def is_in_region(point, region):
    # point (x, y)
    # vertices [(x1, y1), (x2, y2),..., (xn, yn), weight]
    vertices = region[:-1]

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


# Chat gpt wrote this I have no idea how this works
def is_point_on_segment(px, py, p1x, p1y, p2x, p2y):
    # Check if point (px, py) is on the line segment between (p1x, p1y) and (p2x, p2y)
    if min(p1x, p2x) <= px <= max(p1x, p2x) and min(p1y, p2y) <= py <= max(p1y, p2y):
        # Check if the points are collinear using the cross-product method
        return (p2y - p1y) * (px - p1x) == (py - p1y) * (p2x - p1x)
    return False


def calculate_angle(v1, v2):

    v1_magnitude = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    v2_magnitude = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

    costheta = np.dot(v1, v2) / (v1_magnitude * v2_magnitude)
    if costheta > 1:
        costheta = 1

    theta = math.acos(costheta)

    return theta


def calculate_normal(vector):
    x, y = vector
    # Return both perpendicular vectors
    normal1 = (-y, x)
    normal2 = (y, -x)

    return normal1, normal2


def dist_between_points(p1, p2):
    p1x, p1y = p1
    p2x, p2y = p2

    x_squared = (p2x - p1x) ** 2
    y_squared = (p2y - p1y) ** 2

    return math.sqrt(x_squared + y_squared)


# p: position tuple
# vec: vector tuple
# vert1: one vertex on edge
# vert2: second vertex on edge
def find_intersection(p, vec, vert1, vert2):
    # Unpack points
    x1, y1 = p
    x2, y2 = (p[0] + vec[0], p[1] + vec[1])
    x3, y3 = vert1
    x4, y4 = vert2

    # Calculate the coefficients for the line equations
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1

    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3

    # Set up the system of equations in matrix form
    # A1*x + B1*y = C1
    # A2*x + B2*y = C2
    coeff_matrix = np.array([[A1, B1], [A2, B2]])
    const_matrix = np.array([C1, C2])

    # Check if lines are parallel by calculating the determinant
    det = np.linalg.det(coeff_matrix)
    if det == 0:
        return None  # Lines are parallel and do not intersect

    # Solve for x and y
    x, y = np.linalg.solve(coeff_matrix, const_matrix)

    # Check that the intersection is on the correct side of the ray
    if (x - x1) / vec[0] >= 0:
        return (x, y)
    else:
        return None


# p: point (x, y)
# v: vector (a, b)
def get_next(p, v):
    # Find the current region
    region_index = get_current_region(p, v)
    if region_index is None:
        # The point is in no region
        # TODO
        pass
    else:
        # Find the first time the edge of that region is reached

        # Remove the weight from the end of the region
        verts = regions[region_index][:-1]

        n = len(verts)
        for i in range(n):
            # The edge to check uses the current and next vertices
            vert1 = verts[i % n]
            vert2 = verts[(i + 1) % n]

            # Find the intersection coordinates between the point+vector, and the edge
            intersection = find_intersection(p, v, vert1, vert2)

            if intersection is not None:
                # If the edges intersect, find the distance
                dist = dist_between_points(p, intersection)
                # If the distance is 0 it must be checking the edge it's on, so ignore it
                # If the distance is greater than 0, it has found the edge so break
                if dist > 0:
                    break

    # Get the coordinates of the intersection
    # Calculate the normal of the edge of intersection
    # Calculate the angle between the incident ray and the edge
    # Get the cost of the new region
    edge_vector = (vert2[0] - vert1[0], vert2[1] - vert1[1])
    n1, n2 = calculate_normal(edge_vector)

    if np.dot(v, n1) > 0:
        normal = n1
        normal_magnitude = math.sqrt(n1[0] ** 2 + n1[1] ** 2)
        normal = (n1[0] / normal_magnitude, n1[1] / normal_magnitude)
    else:
        normal = n2
        normal_magnitude = math.sqrt(n2[0] ** 2 + n2[1] ** 2)
        normal = (n2[0] / normal_magnitude, n2[1] / normal_magnitude)

    theta = calculate_angle(v, normal)

    det = v[0] * normal[1] - v[1] * normal[0]

    if det < 0:
        theta = -1 * theta

    new_region = get_current_region(intersection, v)
    if new_region == None:
        return intersection, theta, normal, 1
    else:
        new_region_weight = regions[new_region][-1]
        return intersection, theta, normal, new_region_weight


if __name__ == "__main__":

    intersection, theta, normal, new_weight = get_next((0, 0), (math.sqrt(2) / 2, math.sqrt(2) / 2))
    print(
        f"Intersection: {intersection}\nAngle: {theta}\nNormal: {normal}\nNew Weight: {new_weight}"
    )
