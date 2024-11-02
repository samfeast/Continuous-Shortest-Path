import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import random


def main():
    regions = get_example_regions()
    generator = GenerateNodes(regions)
    points = generator.generate_all_points()
    generator.visualize_points()
    print(points)


class GenerateNodes:
    def __init__(self, regions: Dict[int, Tuple[List[Tuple[Tuple[float, float], Tuple[float, float]]], float]]):
        """
        Initialize the node generator.
        
        Args:
            regions: Dictionary mapping region index to (list of line segments, cost)
                    Format: {region_id: ([(start_point, end_point), ...], cost)}
                    Each line segment is defined by two points (x,y)
        """
        self.regions = regions
        self.vertex_cost = {}
        self.nodes = {}
        self.space_bounds = self._calculate_space_bounds()
        self.vertices = self._extract_vertices()
        self.generate_vertex_cost()

    def _calculate_space_bounds(self) -> Tuple[float, float]:
        """
        Calculate the maximum x and y coordinates from all line segments
        """
        max_x = 0
        max_y = 0
        
        for lines, _ in self.regions.values():
            for (start_x, start_y), (end_x, end_y) in lines:
                max_x = max(max_x, start_x, end_x)
                max_y = max(max_y, start_y, end_y)
        
        return (max_x, max_y)

    def _extract_vertices(self) -> Dict[int, List[Tuple[float, float]]]:
        """
        Extract unique vertices for each region from its line segments
        """
        region_vertices = {}
        
        for region_id, (lines, _) in self.regions.items():
            vertices = set()
            for (start_point, end_point) in lines:
                vertices.add(start_point)
                vertices.add(end_point)
            region_vertices[region_id] = list(vertices)
            
        return region_vertices

    def generate_vertex_cost(self):
        """
        Generate vertex ownership based on minimum cost regions
        Vertices are shared between regions where line segments meet
        """
        for region_id, (_, cost) in self.regions.items():
            for vertex in self.vertices[region_id]:
                if vertex not in self.vertex_cost:
                    self.vertex_cost[vertex] = (cost, region_id)
                else:
                    if cost < self.vertex_cost[vertex][0]:
                        self.vertex_cost[vertex] = (cost, region_id)

    def is_point_valid(self, point: Tuple[float, float]) -> bool:
        """
        Check if a point is within the valid space bounds
        """
        x, y = point
        return (x >= 0 and y >= 0 and 
                x <= self.space_bounds[0] and 
                y <= self.space_bounds[1])

    def generate_offset_point(self, 
                            vertex: Tuple[float, float], 
                            lines: List[Tuple[Tuple[float, float], Tuple[float, float]]], 
                            offset: float = 1e-3) -> Tuple[float, float]:
        """
        Generate a point slightly inside the region from a vertex
        """
        
        # Calculate vectors along connected lines
        vectors = []
        for start, end in lines:
            if start == vertex:
                vec = np.array(end) - np.array(vertex)
            else:
                vec = np.array(start) - np.array(vertex)
            vectors.append(vec / np.linalg.norm(vec))
        
        # Calculate bisector (pointing inside)
        bisector = -(vectors[0] + vectors[1])
        if np.linalg.norm(bisector) > 0:
            bisector = bisector / np.linalg.norm(bisector)
            
            # Generate new point
            new_point = tuple(np.array(vertex) + offset * bisector)
            
            # Check if point is within bounds
            if self.is_point_valid(new_point):
                return new_point
        
        return None

    def generate_vertex_points(self, vertex: Tuple[float, float], 
                             lines: List[Tuple[Tuple[float, float], Tuple[float, float]]], 
                             prob: float = 0.5) -> List[Tuple[float, float]]:
        """
        Generate points at vertices with 50% probability
        """
        if random.random() < prob:
            print("Generating node for:", vertex)
            offset_point = self.generate_offset_point(vertex, lines)
            return [offset_point] if offset_point else []
        else:
            print(vertex)
        return []

    def generate_line_points(self, 
                           start: Tuple[float, float], 
                           end: Tuple[float, float], 
                           num_points: int = 3,
                           prob: float = 0.1,
                           offset: float = 1e-3) -> List[Tuple[float, float]]:
        """
        Generate points along a line segment with low probability
        """
        points = []
        start_array = np.array(start)
        end_array = np.array(end)
        
        line_vector = end_array - start_array
        line_length = np.linalg.norm(line_vector)
        if line_length == 0:
            return points
            
        line_vector = line_vector / line_length
        normal_vector = np.array([-line_vector[1], line_vector[0]])
        
        for i in range(1, num_points + 1):
            if random.random() < prob:  # Only generate point with 10% probability
                t = i / (num_points + 1)
                base_point = start_array * (1 - t) + end_array * t
                offset_point = tuple(base_point + normal_vector * offset)
                
                if self.is_point_valid(offset_point):
                    points.append(offset_point)
                
        return points

    def generate_all_points(self) -> Dict[int, List[Tuple[float, float]]]:
        """
        Generate all points for all regions, ensuring shared edges use lower cost
        """
        edge_ownership = {}
        
        # First, determine edge ownership
        for region_id, (lines, cost) in self.regions.items():
            for line in lines:
                sorted_line = tuple(sorted([line[0], line[1]]))
                if sorted_line not in edge_ownership:
                    edge_ownership[sorted_line] = (cost, region_id)
                else:
                    if cost < edge_ownership[sorted_line][0]:
                        edge_ownership[sorted_line] = (cost, region_id)
        
        # Generate points for each region
        for region_id, (lines, cost) in self.regions.items():
            points = []
            
            for vertex in self.vertices[region_id]:
                if self.vertex_cost[vertex][1] == region_id:
                    vertex_points = self.generate_vertex_points(vertex, lines)
                    points.extend(vertex_points)
            
            # Generate edge points
            for line in lines:
                sorted_line = tuple(sorted([line[0], line[1]]))
                if edge_ownership[sorted_line][1] == region_id:  # Only if we own this edge
                    line_points = self.generate_line_points(line[0], line[1])
                    points.extend(line_points)
            
            for point in points:
                self.nodes[region_id] = (points, cost)

        return self.nodes

    def visualize_points(self):
        """
        Visualize regions and vertices with cost-based colors
        """
        plt.figure(figsize=(12, 12))
        
        # Plot bounds
        plt.plot([0, self.space_bounds[0]], [0, 0], 'k--', alpha=0.3)
        plt.plot([0, 0], [0, self.space_bounds[1]], 'k--', alpha=0.3)
        plt.plot([self.space_bounds[0], self.space_bounds[0]], 
                [0, self.space_bounds[1]], 'k--', alpha=0.3)
        plt.plot([0, self.space_bounds[0]], 
                [self.space_bounds[1], self.space_bounds[1]], 'k--', alpha=0.3)
        
        # Get unique costs and create color mapping
        costs = [cost for _, (_, cost) in self.regions.items()]
        unique_costs = sorted(set(costs))
        cost_to_color = {cost: plt.cm.viridis(i/max(1, len(unique_costs)-1)) 
                        for i, cost in enumerate(unique_costs)}
        
        # Plot region boundaries
        for region_id, (lines, cost) in self.regions.items():
            color = cost_to_color[cost]
            for start, end in lines:
                plt.plot([start[0], end[0]], [start[1], end[1]], 
                        '-', color=color, alpha=0.8, linewidth=1,
                        label=f'Cost = {cost}')
        
        # Plot vertices (only at line intersections)
        for vertices, cost in self.nodes.values():
            for vertex in vertices:
                color = cost_to_color[cost]
                plt.plot(vertex[0], vertex[1], 'o', color=color, markersize=4, 
                        markeredgecolor='black', markeredgewidth=0.5)
        
        # Remove duplicate legend entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.axis('equal')
        plt.grid(True)
        plt.xlim(-0.1, self.space_bounds[0] * 1.1)
        plt.ylim(-0.1, self.space_bounds[1] * 1.1)
        plt.show()


def get_example_regions():
    return {
        0: ([((0, 0), (1, 0)), ((1, 0), (1, 1)), ((1, 1), (0, 1)), ((0, 1), (0, 0))], 5),
        1: ([((0, 1), (1, 1)), ((1, 1), (1, 2)), ((1, 2), (0, 2)), ((0, 2), (0, 1))], 3),
        2: ([((0, 2), (1, 2)), ((1, 2), (1, 3)), ((1, 3), (0, 3)), ((0, 3), (0, 2))], 5),
        3: ([((0, 3), (1, 3)), ((1, 3), (1, 4)), ((1, 4), (0, 4)), ((0, 4), (0, 3))], 3),
        4: ([((0, 4), (1, 4)), ((1, 4), (1, 5)), ((1, 5), (0, 5)), ((0, 5), (0, 4))], 3),
        5: ([((1, 0), (2, 0)), ((2, 0), (2, 1)), ((2, 1), (1, 1)), ((1, 1), (1, 0))], 3),
        6: ([((1, 1), (2, 1)), ((2, 1), (2, 2)), ((2, 2), (1, 2)), ((1, 2), (1, 1))], 5),
        7: ([((1, 2), (2, 2)), ((2, 2), (2, 3)), ((2, 3), (1, 3)), ((1, 3), (1, 2))], 1),
        8: ([((1, 3), (2, 3)), ((2, 3), (2, 4)), ((2, 4), (1, 4)), ((1, 4), (1, 3))], 5),
        9: ([((1, 4), (2, 4)), ((2, 4), (2, 5)), ((2, 5), (1, 5)), ((1, 5), (1, 4))], 5),
        10: ([((2, 0), (3, 0)), ((3, 0), (3, 1)), ((3, 1), (2, 1)), ((2, 1), (2, 0))], 5),
        11: ([((2, 1), (3, 1)), ((3, 1), (3, 2)), ((3, 2), (2, 2)), ((2, 2), (2, 1))], 3),
        12: ([((2, 2), (3, 2)), ((3, 2), (3, 3)), ((3, 3), (2, 3)), ((2, 3), (2, 2))], 4),
        13: ([((2, 3), (3, 3)), ((3, 3), (3, 4)), ((3, 4), (2, 4)), ((2, 4), (2, 3))], 1),
        14: ([((2, 4), (3, 4)), ((3, 4), (3, 5)), ((3, 5), (2, 5)), ((2, 4), (2, 5))], 4),
        15: ([((3, 0), (4, 0)), ((4, 0), (4, 1)), ((4, 1), (3, 1)), ((3, 1), (3, 0))], 5),
        16: ([((3, 1), (4, 1)), ((4, 1), (4, 2)), ((4, 2), (3, 2)), ((3, 2), (3, 1))], 4),
        17: ([((3, 2), (4, 2)), ((4, 2), (4, 3)), ((4, 3), (3, 3)), ((3, 2), (3, 3))], 2),
        18: ([((3, 3), (4, 3)), ((4, 3), (4, 4)), ((4, 4), (3, 4)), ((3, 3), (3, 4))], 4),
        19: ([((3, 4), (4, 4)), ((4, 4), (4, 5)), ((4, 5), (3, 5)), ((3, 4), (3, 5))], 5),
        20: ([((4, 0), (5, 0)), ((5, 0), (5, 1)), ((5, 1), (4, 1)), ((4, 1), (4, 0))], 4),
        21: ([((4, 1), (5, 1)), ((5, 1), (5, 2)), ((5, 2), (4, 2)), ((4, 1), (4, 2))], 4),
        22: ([((4, 2), (5, 2)), ((5, 2), (5, 3)), ((5, 3), (4, 3)), ((4, 2), (4, 3))], 3),
        23: ([((4, 3), (5, 3)), ((5, 3), (5, 4)), ((5, 4), (4, 4)), ((4, 3), (4, 4))], 3),
        24: ([((4, 4), (5, 4)), ((5, 4), (5, 5)), ((5, 5), (4, 5)), ((4, 4), (4, 5))], 3)
    }


if __name__ == "__main__":
    main()
