import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt


class generateNodes:
    def __init__(self, polygons: List[Tuple[float, float]]):
        self.polygons = polygons


    def is_point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        """
        Ray casting algorithm to determine if a point is inside a polygon
        """
        x, y = point
        inside = False
        
        j = len(polygon) - 1
        for i in range(len(polygon)):
            if ((polygon[i][1] > y) != (polygon[j][1] > y) and
                x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) /
                    (polygon[j][1] - polygon[i][1]) + polygon[i][0]):
                inside = not inside
            j = i
        
        return inside

    def distance_to_nearest_vertex(point: Tuple[float, float], vertices: List[Tuple[float, float]]) -> float:
        """
        Calculate the minimum distance from a point to any vertex
        """
        return min(np.sqrt((point[0] - v[0])**2 + (point[1] - v[1])**2) for v in vertices)


    def get_polygon_bounds(polygon: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
        """
        Get the bounding box of the polygon
        """
        xs, ys = zip(*polygon)
        return min(xs), max(xs), min(ys), max(ys)


    def calculate_polygon_area(polygon: List[Tuple[float, float]]) -> float:
        """
        Calculate the area of the polygon
        """
        n = len(polygon)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]
        return abs(area) / 2.0


    def generate_region_points(self, base_density: float = 0.1,
                               vertex_density_factor: float = 5.0,
                               vertex_influence_radius: float = None) -> List[Tuple[float, float]]:
        """
        Generate points within a polygon with higher density near vertices.
        
        Args:
            polygon: List of (x, y) tuples defining the polygon vertices
            base_density: Base density of points (points per unit area)
            vertex_density_factor: How much denser the points should be near vertices
            vertex_influence_radius: Radius of increased density around vertices (auto-calculated if None)
        
        Returns:
            List of (x, y) points within the polygon
        """
        # Calculate bounding box
        min_x, max_x, min_y, max_y = get_polygon_bounds(polygon)
        
        # Calculate area and auto-determine vertex influence radius if not specified
        area = calculate_polygon_area(polygon)
        if vertex_influence_radius is None:
            vertex_influence_radius = np.sqrt(area) * 0.15  # 15% of square root of area
        
        # Calculate number of points based on area and density
        target_points = int(area * base_density)
        
        # Generate more points than needed (we'll filter some out)
        points = []
        attempts = 0
        max_attempts = target_points * 10
        
        while len(points) < target_points and attempts < max_attempts:
            # Generate random point within bounding box
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            point = (x, y)
            
            # Check if point is in polygon
            if not is_point_in_polygon(point, polygon):
                attempts += 1
                continue
                
            # Calculate distance to nearest vertex
            dist = distance_to_nearest_vertex(point, polygon)
            
            # Probability of keeping the point increases near vertices
            prob = 1.0
            if dist < vertex_influence_radius:
                # Smoothly increase probability near vertices
                prob = 1.0 + (vertex_density_factor - 1.0) * (1.0 - dist/vertex_influence_radius)**2
            
            if np.random.random() < prob:
                points.append(point)
                
            attempts += 1
        
        return points


    def visualize_points(polygon: List[Tuple[float, float]], points: List[Tuple[float, float]]):
        """
        Visualize the polygon and generated points using matplotlib
        """
        
        # Plot polygon
        polygon_plus_first = polygon + [polygon[0]]  # Close the polygon
        xs, ys = zip(*polygon_plus_first)
        plt.plot(xs, ys, 'b-', label='Polygon')
        
        # Plot points
        point_xs, point_ys = zip(*points)
        plt.scatter(point_xs, point_ys, c='r', s=10, alpha=0.5, label='Generated Points')
        
        plt.axis('equal')
        plt.legend()
        plt.grid(True)
        plt.show()
