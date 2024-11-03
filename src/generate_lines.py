import numpy as np
import math

EPSILON = (0.00001, 0.00001)


class Graph:

    def __init__(self, regions, target):
        self.regions = regions
        self.target = target

        self.current_region = None
        self.start_region = None
        self.adjacency = {}

        self.add_bounding_regions(target)
        self.compute_region_adjacency(regions)

    def add_bounding_regions(self, target):
        x, y = target

        bbox_1 = [(-1, 1), (x + 1, 1), (x + 1, 0), (-1, 0), -1]
        bbox_2 = [(-1, 0), (0, 0), (0, y), (-1, y), -1]
        bbox_3 = [(-1, y), (x + 1, y), (x + 1, y - 1), (-1, y - 1), -1]
        bbox_4 = [(x, 0), (x + 1, 0), (x + 1, y), (x, y), -1]

        self.regions.append(bbox_1)
        self.regions.append(bbox_2)
        self.regions.append(bbox_3)
        self.regions.append(bbox_4)

    def compute_region_adjacency(self, regions):
        all_region_edges = []
        i = 0
        for region in regions:
            self.adjacency[i] = []
            region = region[:-1]
            n = len(region)
            region_edges = []
            for j in range(n):
                region_edges.append((region[j], region[(j + 1) % n]))
            all_region_edges.append(region_edges)
            i += 1

        n = len(regions)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                for edge_i in all_region_edges[i]:
                    v1i, v2i = edge_i
                    for edge_j in all_region_edges[j]:
                        if j in self.adjacency[i]:
                            continue

                        v1j, v2j = edge_j

                        if v2i[0] == v1i[0]:
                            edge_i_gradient = float("inf")
                        else:
                            edge_i_gradient = (v2i[1] - v1i[1]) / (v2i[0] - v1i[0])

                        if v2j[0] == v1j[0]:
                            edge_j_gradient = float("inf")
                        else:
                            edge_j_gradient = (v2j[1] - v1j[1]) / (v2j[0] - v1j[0])

                        # Check if the two lines have the same gradient (accounting for inaccuracy)
                        if (
                            edge_i_gradient <= edge_j_gradient + EPSILON[0]
                            and edge_i_gradient >= edge_j_gradient - EPSILON[0]
                        ):
                            if self.lies_in_interval(v1i, v2i, v1j) or self.lies_in_interval(
                                v1i, v2i, v2j
                            ):
                                # If one of the vertices of j lies strictly inside i
                                self.adjacency[i].append(j)
                                continue
                            elif self.lies_in_interval(v1j, v2j, v1i) or self.lies_in_interval(
                                v1j, v2j, v2i
                            ):
                                # If one of the vertices of i lies strictly inside j
                                self.adjacency[i].append(j)
                                continue
                            elif set(edge_i) == set(edge_j):
                                # If i and j are the same edge
                                self.adjacency[i].append(j)
                                continue

    def get_regions(self):
        return self.regions

    def get_target(self):
        return self.target

    def get_adjacency(self):
        return self.adjacency

    def lies_in_interval(self, vert1, vert2, point):
        v1x, v1y = vert1
        v2x, v2y = vert2
        x, y = point

        if min(v1x, v2x) <= x and x <= max(v1x, v2x):
            if min(v1y, v2y) <= y and y <= max(v1y, v2y):
                return True

        return False

    def reset_current_region(self):
        self.current_region = None

    # Get the current region based on a point and vector
    # This is to be used when the ray is at an edge
    def get_current_region(self, point, vector):

        # Nudge the position slightly in the direction it's pointing
        point = (point[0] + vector[0] * EPSILON[0], point[1] + vector[1] * EPSILON[1])
        # If there is no current region stored check all regions
        if self.current_region is None:
            i = 0
            # Check all the regions, return when the point is in one
            for region in self.regions:
                if self.is_in_region(point, region):
                    self.current_region = i
                    return i
                i += 1
        else:

            # Check if its not moved to a new region
            if self.is_in_region(point, self.regions[self.current_region]):
                return self.current_region

            n = len(self.regions)
            # If it's moved to a new region, then go through all regions
            for i in range(n):
                # If region i is adjacent to the current region, check if we're now in that region
                if i in self.adjacency[self.current_region]:
                    if self.is_in_region(point, self.regions[i]):
                        self.current_region = i
                        return i

        return None

    def is_in_region(self, point, region):
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
            if self.is_point_on_segment(x, y, p1x, p1y, p2x, p2y):
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

    def is_strictly_in_region(self, point, region):
        # point (x, y)
        # vertices [(x1, y1), (x2, y2),..., (xn, yn), weight]
        vertices = region[:-1]

        x, y = point
        n = len(vertices)
        inside = False

        p1x, p1y = vertices[0]
        for i in range(n + 1):
            p2x, p2y = vertices[i % n]

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
    def is_point_on_segment(self, px, py, p1x, p1y, p2x, p2y):
        # Check if point (px, py) is on the line segment between (p1x, p1y) and (p2x, p2y)
        if min(p1x, p2x) <= px <= max(p1x, p2x) and min(p1y, p2y) <= py <= max(p1y, p2y):
            # Check if the points are collinear using the cross-product method
            return (p2y - p1y) * (px - p1x) == (py - p1y) * (p2x - p1x)
        return False

    def calculate_angle(self, v1, v2):

        v1_magnitude = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        v2_magnitude = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        costheta = np.dot(v1, v2) / (v1_magnitude * v2_magnitude)
        if costheta > 1:
            costheta = 1

        theta = math.acos(costheta)

        return theta

    def calculate_normal(self, vector):
        x, y = vector
        # Return both perpendicular vectors
        normal1 = (-y, x)
        normal2 = (y, -x)

        return normal1, normal2

    def dist_between_points(self, p1, p2):
        p1x, p1y = p1
        p2x, p2y = p2

        x_squared = (p2x - p1x) ** 2
        y_squared = (p2y - p1y) ** 2

        return math.sqrt(x_squared + y_squared)

    # p: position tuple
    # vec: vector tuple
    # vert1: one vertex on edge
    # vert2: second vertex on edge
    def find_intersection(self, p, v, vert1, vert2):
        # Unpack points
        x1, y1 = p
        x2, y2 = (p[0] + v[0], p[1] + v[1])
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
        if self.lies_in_interval(vert1, vert2, (x, y)):
            return (x, y)
        else:
            return None

    # p: point (x, y)
    # v: vector (a, b)
    def get_next(self, p, v):
        # Find the current region
        region_index = self.get_current_region(p, v)
        if region_index is None:
            min_dist = float("inf")
            min_vert1 = None
            min_vert2 = None
            min_intersection = None
            for region in self.regions:
                verts = region[:-1]
                n = len(verts)
                for i in range(n):

                    # Find the first time the edge of that region is reached

                    # Remove the weight from the end of the region

                    # The edge to check uses the current and next vertices
                    vert1 = verts[i % n]
                    vert2 = verts[(i + 1) % n]

                    # Find the intersection coordinates between the point+vector, and the edge
                    intersection = self.find_intersection(p, v, vert1, vert2)

                    if intersection is not None:
                        # If the edges intersect, find the distance
                        dist = self.dist_between_points(p, intersection)
                        # If the distance is 0 it must be checking the edge it's on, so ignore it
                        # If the distance is greater than 0, it has found the edge so break
                        if dist > 0 and dist < min_dist:
                            min_dist = dist
                            min_vert1 = vert1
                            min_vert2 = vert2
                            min_intersection = intersection
        else:
            # Find the first time the edge of that region is reached

            # Remove the weight from the end of the region
            verts = self.regions[region_index][:-1]

            n = len(verts)
            min_dist = float("inf")
            min_vert1 = None
            min_vert2 = None
            min_intersection = None
            for i in range(n):
                # The edge to check uses the current and next vertices
                vert1 = verts[i % n]
                vert2 = verts[(i + 1) % n]

                # Find the intersection coordinates between the point+vector, and the edge
                intersection = self.find_intersection(p, v, vert1, vert2)

                if intersection is not None:
                    # If the edges intersect, find the distance
                    dist = self.dist_between_points(p, intersection)
                    # If the distance is 0 it must be checking the edge it's on, so ignore it
                    # If the distance is greater than 0, it has found the edge so break
                    if dist > 0 and dist < min_dist:
                        min_dist = dist
                        min_vert1 = vert1
                        min_vert2 = vert2
                        min_intersection = intersection

        # Get the coordinates of the intersection
        # Calculate the normal of the edge of intersection
        # Calculate the angle between the incident ray and the edge
        # Get the cost of the new region
        edge_vector = (min_vert2[0] - min_vert1[0], min_vert2[1] - min_vert1[1])
        n1, n2 = self.calculate_normal(edge_vector)

        if np.dot(v, n1) > 0:
            normal = n1
            normal_magnitude = math.sqrt(n1[0] ** 2 + n1[1] ** 2)
            normal = (n1[0] / normal_magnitude, n1[1] / normal_magnitude)
        else:
            normal = n2
            normal_magnitude = math.sqrt(n2[0] ** 2 + n2[1] ** 2)
            normal = (n2[0] / normal_magnitude, n2[1] / normal_magnitude)

        theta = self.calculate_angle(v, normal)

        det = v[0] * normal[1] - v[1] * normal[0]

        if det < 0:
            theta = -1 * theta

        new_region = self.get_current_region(min_intersection, v)
        if new_region == None:
            min_intersection = (
                min_intersection[0] + normal[0] * EPSILON[0],
                min_intersection[1] + normal[1] * EPSILON[1],
            )
            return min_intersection, theta, normal, 1
        else:
            new_region_weight = self.regions[new_region][-1]
            min_intersection = (
                min_intersection[0] - normal[0] * EPSILON[0],
                min_intersection[1] - normal[1] * EPSILON[1],
            )
            return min_intersection, theta, normal, new_region_weight


if __name__ == "__main__":

    my_regions = [[(1 / 3 + 0.1, -1 / 3), (2 / 3, -1 / 3), (1 / 2, -1 / 2), 3]]
    my_target = (1, -1)
    graph = Graph([], my_target)

    intersection, theta, normal, new_weight = graph.get_next((0, 0), (0.5, -0.5))
    print(
        f"Intersection: {intersection}\nAngle: {theta}\nNormal: {normal}\nNew Weight: {new_weight}"
    )
