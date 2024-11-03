from nodes import GenerateNodes
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import heapq


class Dijkstra:
    """
    Implementation of Dijkstra's algorithm for finding shortest paths between points
    in a graph where nodes are constrained to regions with different travel costs.
    """
    
    def __init__(self, graph_adjacency: Dict[int, Set[int]], nodes: Dict[int, Tuple[List[Tuple[float, float]], float]]):
        """
        Initialize the Dijkstra algorithm with graph structure and node information.
        
        Args:
            graph_adjacency: Dictionary mapping region ID to set of adjacent region IDs
            nodes: Dictionary mapping region ID to tuple of (points, cost)
                  where points is list of (x,y) coordinates and cost is float
        """
        self.graph_adjacency = graph_adjacency
        self.nodes = nodes
        self.region_points = self._build_region_points()
        # Distance cache
        self.distance_cache = {}

    def _build_region_points(self) -> Dict[Tuple[float, float], int]:
        """
        Build mapping from points to their corresponding regions.
        
        Returns:
            Dictionary mapping point coordinates to region ID
        """
        region_points = {}
        for region_id, (points, _) in self.nodes.items():
            for point in points:
                region_points[point] = region_id
        return region_points
    
    def _get_point_region(self, point: Tuple[float, float]) -> int:
        """
        Get the region ID for a given point.
        
        Args:
            point: Tuple of (x,y) coordinates
            
        Returns:
            Region ID containing the point
        """
        return self.region_points[point]
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Cached distance calculation"""
        cache_key = (min(point1, point2), max(point1, point2))
        if cache_key not in self.distance_cache:
            self.distance_cache[cache_key] = np.sqrt((point1[0] - point2[0])**2 + 
                                                    (point1[1] - point2[1])**2)
        return self.distance_cache[cache_key]
    
    def _calculate_travel_cost(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """
        Calculate travel cost between two points considering their regions.
        
        Args:
            point1: Starting point coordinates
            point2: Ending point coordinates
            
        Returns:
            Travel cost between points
        """
        region1 = self._get_point_region(point1)
        region2 = self._get_point_region(point2)
        distance = self._calculate_distance(point1, point2)
        
        if region1 == region2:
            # Within same region
            return distance * self.nodes[region1][1]
        elif region2 in self.graph_adjacency[region1]:  # Check adjacency using dictionary
            # Adjacent regions
            avg_cost = (self.nodes[region1][1] + self.nodes[region2][1]) / 2
            return distance * avg_cost
        else:
            return float('inf')
    
    def _get_neighbors(self, point: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Get neighboring points from current and adjacent regions"""
        current_region = self._get_point_region(point)
        neighbors = []
        
        # Add points from same region
        neighbors.extend(p for p in self.nodes[current_region][0] if p != point)
        
        # Add points from adjacent regions
        for region_id in self.graph_adjacency[current_region]:
            neighbors.extend(self.nodes[region_id][0])
                
        return neighbors
    
    def find_shortest_path(self, start: Tuple[float, float], target: Tuple[float, float]) -> Tuple[List[Tuple[float, float]], float]:
        """
        Find shortest path from start to target point using Dijkstra's algorithm.
        
        Args:
            start: Starting point coordinates
            target: Target point coordinates
            
        Returns:
            Tuple of (path, total_cost) where path is list of points and total_cost is float
        """
        if start not in self.region_points or target not in self.region_points:
            raise ValueError("Start or target point not found in any region")
            
        # Initialize distances and previous points
        distances = defaultdict(lambda: float('inf'))
        distances[start] = 0
        previous = {}
        
        # Priority queue for Dijkstra's algorithm
        pq = [(0, start)]
        visited = set()
        
        max_possible_optimal = float('inf')
        while pq:
            current_distance, current_point = heapq.heappop(pq)
            
            # Early termination check
            if current_distance > max_possible_optimal:
                break
                
            if current_point == target:
                max_possible_optimal = current_distance
                continue
                
            if current_point in visited:
                continue
                
            visited.add(current_point)
                
            for neighbor in self._get_neighbors(current_point):
                travel_cost = self._calculate_travel_cost(current_point, neighbor)
                distance = current_distance + travel_cost
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current_point
                    heapq.heappush(pq, (distance, neighbor))
        
        # Reconstruct path
        if target not in previous and start != target:
            return [], float('inf')
            
        path = []
        current = target
        while current in previous:
            path.append(current)
            current = previous[current]
        path.append(start)
        path.reverse()
        
        return path, distances[target]


class AStar:
    """
    Implementation of A* algorithm for finding shortest paths between points
    in a graph where nodes are constrained to regions with different travel costs.
    """
    
    def __init__(self, graph_adjacency: np.ndarray, nodes: Dict[int, Tuple[List[Tuple[float, float]], float]]):
        """
        Initialize the A* algorithm with graph structure and node information.
        
        Args:
            graph_adjacency: nxn matrix where entry (i,j) is 1 if regions i and j are adjacent
            nodes: Dictionary mapping region ID to tuple of (points, cost)
                  where points is list of (x,y) coordinates and cost is float
        """
        self.graph_adjacency = graph_adjacency
        self.nodes = nodes
        self.region_points = self._build_region_points()
        self.min_cost = min(cost for _, cost in nodes.values())
        
    def _build_region_points(self) -> Dict[Tuple[float, float], int]:
        """
        Build mapping from points to their corresponding regions.
        
        Returns:
            Dictionary mapping point coordinates to region ID
        """
        region_points = {}
        for region_id, (points, _) in self.nodes.items():
            for point in points:
                region_points[point] = region_id
        return region_points
    
    def _get_point_region(self, point: Tuple[float, float]) -> int:
        """
        Get the region ID for a given point.
        
        Args:
            point: Tuple of (x,y) coordinates
            
        Returns:
            Region ID containing the point
        """
        return self.region_points[point]
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: First point coordinates (x1, y1)
            point2: Second point coordinates (x2, y2)
            
        Returns:
            Euclidean distance between points
        """
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _calculate_travel_cost(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """
        Calculate travel cost between two points considering their regions.
        
        Args:
            point1: Starting point coordinates
            point2: Ending point coordinates
            
        Returns:
            Travel cost between points
        """
        region1 = self._get_point_region(point1)
        region2 = self._get_point_region(point2)
        distance = self._calculate_distance(point1, point2)
        
        if region1 == region2:
            # Within same region
            return distance * self.nodes[region1][1]
        elif self.graph_adjacency[region1][region2] == 1:
            # Adjacent regions
            avg_cost = (self.nodes[region1][1] + self.nodes[region2][1]) / 2
            return distance * avg_cost
        else:
            return float('inf')
    
    def _get_neighbors(self, point: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Get all possible neighboring points that can be reached from current point.
        
        Args:
            point: Current point coordinates
            
        Returns:
            List of neighboring point coordinates
        """
        current_region = self._get_point_region(point)
        neighbors = []
        
        # Add points from same region
        for p in self.nodes[current_region][0]:
            if p != point:
                neighbors.append(p)
        
        # Add points from adjacent regions
        for region_id in range(len(self.graph_adjacency)):
            if self.graph_adjacency[current_region][region_id] == 1:
                neighbors.extend(self.nodes[region_id][0])
                
        return neighbors
    
    def _heuristic(self, point: Tuple[float, float], target: Tuple[float, float]) -> float:
        """
        Calculate heuristic estimate of cost from point to target.
        Uses minimum possible cost (straight line distance * minimum region cost).
        
        Args:
            point: Current point coordinates
            target: Target point coordinates
            
        Returns:
            Heuristic estimate of minimum possible cost to target
        """
        return self._calculate_distance(point, target) * self.min_cost
    
    def find_shortest_path(self, start: Tuple[float, float], target: Tuple[float, float]) -> Tuple[List[Tuple[float, float]], float]:
        """
        Find shortest path from start to target point using A* algorithm.
        
        Args:
            start: Starting point coordinates
            target: Target point coordinates
            
        Returns:
            Tuple of (path, total_cost) where path is list of points and total_cost is float
        """
        if start not in self.region_points or target not in self.region_points:
            raise ValueError("Start or target point not found in any region")
            
        # Initialize data structures
        g_score = defaultdict(lambda: float('inf'))  # Cost from start to current node
        g_score[start] = 0
        f_score = defaultdict(lambda: float('inf'))  # Estimated total cost through node
        f_score[start] = self._heuristic(start, target)
        
        # Priority queue entries are (f_score, g_score, point)
        # g_score is used as tiebreaker when f_scores are equal
        open_set = [(f_score[start], 0, start)]
        previous = {}
        closed_set = set()
        
        while open_set:
            current_f, current_g, current_point = heapq.heappop(open_set)
            
            if current_point in closed_set:
                continue
                
            if current_point == target:
                break
                
            closed_set.add(current_point)
            
            for neighbor in self._get_neighbors(current_point):
                if neighbor in closed_set:
                    continue
                    
                travel_cost = self._calculate_travel_cost(current_point, neighbor)
                tentative_g = current_g + travel_cost
                
                if tentative_g < g_score[neighbor]:
                    previous[neighbor] = current_point
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, target)
                    heapq.heappush(open_set, (f_score[neighbor], g_score[neighbor], neighbor))
        
        # Reconstruct path
        if target not in previous and start != target:
            return [], float('inf')
            
        path = []
        current = target
        while current in previous:
            path.append(current)
            current = previous[current]
        path.append(start)
        path.reverse()
        
        return path, g_score[target]

    def verify_path(self, path: List[Tuple[float, float]]) -> bool:
        """
        Verify that a path is valid according to region constraints.
        
        Args:
            path: List of points representing the path
            
        Returns:
            True if path is valid, False otherwise
        """
        if not path:
            return True
            
        for i in range(len(path) - 1):
            point1, point2 = path[i], path[i + 1]
            region1 = self._get_point_region(point1)
            region2 = self._get_point_region(point2)
            
            if region1 != region2 and self.graph_adjacency[region1][region2] != 1:
                return False
                
        return True


class BidirectionalSearch:
    """
    Implementation of Bidirectional Search algorithm for finding shortest paths between points
    in a graph where nodes are constrained to regions with different travel costs.
    """
    
    def __init__(self, graph_adjacency: np.ndarray, nodes: Dict[int, Tuple[List[Tuple[float, float]], float]]):
        """
        Initialize the Bidirectional Search algorithm with graph structure and node information.
        
        Args:
            graph_adjacency: nxn matrix where entry (i,j) is 1 if regions i and j are adjacent
            nodes: Dictionary mapping region ID to tuple of (points, cost)
                  where points is list of (x,y) coordinates and cost is float
        """
        self.graph_adjacency = graph_adjacency
        self.nodes = nodes
        self.region_points = self._build_region_points()
        
    def _build_region_points(self) -> Dict[Tuple[float, float], int]:
        """
        Build mapping from points to their corresponding regions.
        
        Returns:
            Dictionary mapping point coordinates to region ID
        """
        region_points = {}
        for region_id, (points, _) in self.nodes.items():
            for point in points:
                region_points[point] = region_id
        return region_points
    
    def _get_point_region(self, point: Tuple[float, float]) -> int:
        """
        Get the region ID for a given point.
        
        Args:
            point: Tuple of (x,y) coordinates
            
        Returns:
            Region ID containing the point
        """
        return self.region_points[point]
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: First point coordinates (x1, y1)
            point2: Second point coordinates (x2, y2)
            
        Returns:
            Euclidean distance between points
        """
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _calculate_travel_cost(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """
        Calculate travel cost between two points considering their regions.
        
        Args:
            point1: Starting point coordinates
            point2: Ending point coordinates
            
        Returns:
            Travel cost between points
        """
        region1 = self._get_point_region(point1)
        region2 = self._get_point_region(point2)
        distance = self._calculate_distance(point1, point2)
        
        if region1 == region2:
            # Within same region
            return distance * self.nodes[region1][1]
        elif self.graph_adjacency[region1][region2] == 1:
            # Adjacent regions
            avg_cost = (self.nodes[region1][1] + self.nodes[region2][1]) / 2
            return distance * avg_cost
        else:
            return float('inf')
    
    def _get_neighbors(self, point: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Get all possible neighboring points that can be reached from current point.
        
        Args:
            point: Current point coordinates
            
        Returns:
            List of neighboring point coordinates
        """
        current_region = self._get_point_region(point)
        neighbors = []
        
        # Add points from same region
        for p in self.nodes[current_region][0]:
            if p != point:
                neighbors.append(p)
        
        # Add points from adjacent regions
        for region_id in range(len(self.graph_adjacency)):
            if self.graph_adjacency[current_region][region_id] == 1:
                neighbors.extend(self.nodes[region_id][0])
                
        return neighbors
    
    def _construct_path(self, 
                       intersection_point: Tuple[float, float],
                       forward_previous: Dict[Tuple[float, float], Tuple[float, float]],
                       backward_previous: Dict[Tuple[float, float], Tuple[float, float]],
                       start: Tuple[float, float],
                       target: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Construct the complete path from start to target through the intersection point.
        
        Args:
            intersection_point: Point where forward and backward searches met
            forward_previous: Path tracking for forward search
            backward_previous: Path tracking for backward search
            start: Starting point coordinates
            target: Target point coordinates
            
        Returns:
            Complete path from start to target
        """
        # Construct path from start to intersection
        forward_path = []
        current = intersection_point
        while current in forward_previous:
            forward_path.append(current)
            current = forward_previous[current]
        forward_path.append(start)
        forward_path.reverse()
        
        # Construct path from intersection to target
        backward_path = []
        current = intersection_point
        while current in backward_previous:
            current = backward_previous[current]
            backward_path.append(current)
            
        # Combine paths
        return forward_path + backward_path
    
    def find_shortest_path(self, start: Tuple[float, float], target: Tuple[float, float]) -> Tuple[List[Tuple[float, float]], float]:
        """
        Find shortest path from start to target point using Bidirectional Search.
        
        Args:
            start: Starting point coordinates
            target: Target point coordinates
            
        Returns:
            Tuple of (path, total_cost) where path is list of points and total_cost is float
        """
        if start not in self.region_points or target not in self.region_points:
            raise ValueError("Start or target point not found in any region")
            
        # Initialize forward search
        forward_distances = defaultdict(lambda: float('inf'))
        forward_distances[start] = 0
        forward_pq = [(0, start)]
        forward_previous = {}
        forward_visited = set()
        
        # Initialize backward search
        backward_distances = defaultdict(lambda: float('inf'))
        backward_distances[target] = 0
        backward_pq = [(0, target)]
        backward_previous = {}
        backward_visited = set()
        
        # Best path information
        best_total_distance = float('inf')
        best_intersection_point = None
        
        while forward_pq and backward_pq:
            # Process forward search
            current_forward_dist, current_forward = heapq.heappop(forward_pq)
            
            if current_forward_dist > best_total_distance:
                break
                
            if current_forward in forward_visited:
                continue
                
            forward_visited.add(current_forward)
            
            # Process backward search
            current_backward_dist, current_backward = heapq.heappop(backward_pq)
            
            if current_backward_dist > best_total_distance:
                break
                
            if current_backward in backward_visited:
                continue
                
            backward_visited.add(current_backward)
            
            # Check for intersection and update best path
            for point in forward_visited:
                if point in backward_visited:
                    total_distance = forward_distances[point] + backward_distances[point]
                    if total_distance < best_total_distance:
                        best_total_distance = total_distance
                        best_intersection_point = point
            
            # Expand forward search
            for neighbor in self._get_neighbors(current_forward):
                if neighbor in forward_visited:
                    continue
                    
                travel_cost = self._calculate_travel_cost(current_forward, neighbor)
                distance = current_forward_dist + travel_cost
                
                if distance < forward_distances[neighbor]:
                    forward_distances[neighbor] = distance
                    forward_previous[neighbor] = current_forward
                    heapq.heappush(forward_pq, (distance, neighbor))
            
            # Expand backward search
            for neighbor in self._get_neighbors(current_backward):
                if neighbor in backward_visited:
                    continue
                    
                travel_cost = self._calculate_travel_cost(current_backward, neighbor)
                distance = current_backward_dist + travel_cost
                
                if distance < backward_distances[neighbor]:
                    backward_distances[neighbor] = distance
                    backward_previous[neighbor] = current_backward
                    heapq.heappush(backward_pq, (distance, neighbor))
        
        if best_intersection_point is None:
            return [], float('inf')
            
        path = self._construct_path(best_intersection_point, forward_previous, 
                                  backward_previous, start, target)
        
        return path, best_total_distance
    
    def verify_path(self, path: List[Tuple[float, float]]) -> bool:
        """
        Verify that a path is valid according to region constraints.
        
        Args:
            path: List of points representing the path
            
        Returns:
            True if path is valid, False otherwise
        """
        if not path:
            return True
            
        for i in range(len(path) - 1):
            point1, point2 = path[i], path[i + 1]
            region1 = self._get_point_region(point1)
            region2 = self._get_point_region(point2)
            
            if region1 != region2 and self.graph_adjacency[region1][region2] != 1:
                return False
                
        return True


if __name__ == "__main__":
    # graph_adjacency = generate_adjacency()
    generator = GenerateNodes()

    # djikstra = Djikstra()