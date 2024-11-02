import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt


class GenerateNodes:
    def __init__(self, polygons: Dict[int, List[Tuple[float, float]]]):
        """
        Initialize the node generator.
        
        Args:
            polygons: Dictionary mapping polygon index to list of vertices and cost
                     Format: {polygon_id: ([vertices], cost)}
        """
        self.polygons = polygons
        self.vertex_cost = {}
        self.generate_vertex_cost()
        self.nodes = set()

    def generate_vertex_cost(self):
        """
        Generate vertex ownership based on minimum cost polygons
        """
        for poly_id, (vertices, cost) in self.polygons.items():
            for vertex in vertices:
                if vertex not in self.vertex_cost:
                    self.vertex_cost[vertex] = (cost, poly_id)
                else:
                    if cost < self.vertex_cost[vertex][0]:
                        self.vertex_cost[vertex] = (cost, poly_id)

    def generate_offset_point(self, point: Tuple[float, float], 
                            polygon: List[Tuple[float, float]], 
                            offset: float = 1e-3) -> Tuple[float, float]:
        """
        Generate a point slightly inside the polygon from a vertex
        """
        # Find adjacent vertices
        idx = polygon.index(point)
        prev_idx = (idx - 1) % len(polygon)
        next_idx = (idx + 1) % len(polygon)
        
        # Get vectors to adjacent vertices
        v1 = np.array(polygon[prev_idx]) - np.array(point)
        v2 = np.array(polygon[next_idx]) - np.array(point)
        
        # Normalize vectors
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Get bisector vector (pointing inside polygon)
        bisector = -(v1 + v2)
        bisector = bisector / np.linalg.norm(bisector)
        
        # Generate new point offset along bisector
        new_point = np.array(point) + offset * bisector
        return tuple(new_point)

    def generate_edge_points(self, polygon: List[Tuple[float, float]], 
                           num_points: int = 3) -> List[Tuple[float, float]]:
        """
        Generate points along the edges of the polygon
        """
        edge_points = []
        for i in range(len(polygon)):
            p1 = np.array(polygon[i])
            p2 = np.array(polygon[(i + 1) % len(polygon)])
            
            # Generate points along edge
            for j in range(1, num_points + 1):
                t = j / (num_points + 1)
                point = p1 * (1 - t) + p2 * t
                
                # Offset point slightly inside polygon
                edge_vector = p2 - p1
                normal_vector = np.array([-edge_vector[1], edge_vector[0]])
                normal_vector = normal_vector / np.linalg.norm(normal_vector)
                
                # Check if normal points inside polygon using cross product
                center = np.mean(polygon, axis=0)
                to_center = center - point
                if np.cross(edge_vector, to_center) < 0:
                    normal_vector = -normal_vector
                    
                offset_point = point + normal_vector * 1e-3
                edge_points.append(tuple(offset_point))
                
        return edge_points

    def generate_interior_points(self, polygon: List[Tuple[float, float]], 
                               num_points: int = 5) -> List[Tuple[float, float]]:
        """
        Generate sparse points in the interior of the polygon
        """
        # Calculate centroid
        centroid = np.mean(polygon, axis=0)
        
        # Generate random points in polygon interior
        interior_points = []
        for _ in range(num_points):
            # Generate random point between centroid and random edge point
            edge_point = polygon[np.random.randint(len(polygon))]
            t = np.random.uniform(0.2, 0.8)  # Bias towards edges
            point = tuple(centroid * (1 - t) + np.array(edge_point) * t)
            interior_points.append(point)
            
        return interior_points

    def generate_all_points(self) -> Dict[int, List[Tuple[float, float]]]:
        """
        Generate all points for all polygons
        """
        polygon_points = {}
        
        # Process polygons in order of increasing cost
        sorted_polygons = sorted(self.polygons.items(), key=lambda x: x[1][1])
        
        for poly_id, (vertices, cost) in sorted_polygons:
            points = []
            
            # Generate vertex-based points
            for vertex in vertices:
                if self.vertex_cost[vertex][1] == poly_id:  # Only if we own this vertex
                    offset_point = self.generate_offset_point(vertex, vertices)
                    points.append(offset_point)
            
            # Generate edge points
            edge_points = self.generate_edge_points(vertices)
            points.extend(edge_points)
            
            # Generate interior points
            interior_points = self.generate_interior_points(vertices)
            points.extend(interior_points)
            
            polygon_points[poly_id] = points
            
        return polygon_points

    def visualize_points(self):
        """
        Visualize all polygons and their generated points
        """
        plt.figure(figsize=(10, 10))
        
        # Plot each polygon and its points
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.polygons)))
        
        polygon_points = self.generate_all_points()
        
        for (poly_id, (vertices, cost)), color in zip(self.polygons.items(), colors):
            # Plot polygon
            vertices_plot = vertices + [vertices[0]]  # Close the polygon
            xs, ys = zip(*vertices_plot)
            plt.plot(xs, ys, '-', color=color, alpha=0.5, label=f'Polygon {poly_id} (cost={cost})')
            
            # Plot points
            if poly_id in polygon_points:
                xs, ys = zip(*polygon_points[poly_id])
                plt.scatter(xs, ys, color=color, s=30)
        
        plt.axis('equal')
        plt.legend()
        plt.grid(True)
        plt.show()
