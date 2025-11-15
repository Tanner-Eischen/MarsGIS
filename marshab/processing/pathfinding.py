"""Path planning algorithms for rover navigation."""

from heapq import heappush, heappop
from typing import List, Tuple, Optional

import numpy as np

from marshab.exceptions import NavigationError
from marshab.utils.logging import get_logger

logger = get_logger(__name__)


class AStarPathfinder:
    """A* pathfinding algorithm for rover navigation."""
    
    def __init__(self, cost_map: np.ndarray, cell_size_m: float = 1.0):
        """Initialize A* pathfinder.
        
        Args:
            cost_map: 2D array of traversability costs
            cell_size_m: Size of each cell in meters
        """
        self.cost_map = cost_map
        self.cell_size_m = cell_size_m
        self.height, self.width = cost_map.shape
    
    def heuristic(
        self,
        a: Tuple[int, int],
        b: Tuple[int, int]
    ) -> float:
        """Calculate heuristic distance (Euclidean).
        
        Args:
            a: Start position (row, col)
            b: Goal position (row, col)
        
        Returns:
            Estimated distance
        """
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2) * self.cell_size_m
    
    def get_neighbors(
        self,
        pos: Tuple[int, int]
    ) -> List[Tuple[int, int, float]]:
        """Get valid neighbors of a position.
        
        Args:
            pos: Current position (row, col)
        
        Returns:
            List of (row, col, cost) tuples for valid neighbors
        """
        row, col = pos
        neighbors = []
        
        # 8-connected grid
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for dr, dc in directions:
            new_row = row + dr
            new_col = col + dc
            
            # Check bounds
            if not (0 <= new_row < self.height and 0 <= new_col < self.width):
                continue
            
            # Check if passable
            cell_cost = self.cost_map[new_row, new_col]
            if np.isinf(cell_cost):
                continue
            
            # Calculate move cost (diagonal moves cost more)
            # Diagonal: both dr and dc are non-zero
            is_diagonal = dr != 0 and dc != 0
            move_cost = self.cell_size_m * (1.414 if is_diagonal else 1.0)
            total_cost = cell_cost * move_cost
            
            neighbors.append((new_row, new_col, total_cost))
        
        return neighbors
    
    def find_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """Find optimal path from start to goal using A*.
        
        Args:
            start: Start position (row, col)
            goal: Goal position (row, col)
        
        Returns:
            List of (row, col) positions forming path, or None if no path exists
        """
        logger.info(
            "Starting A* pathfinding",
            start=start,
            goal=goal,
            cost_map_shape=self.cost_map.shape
        )
        
        # Validate start and goal
        if not (0 <= start[0] < self.height and 0 <= start[1] < self.width):
            raise NavigationError("Start position out of bounds")
        
        if not (0 <= goal[0] < self.height and 0 <= goal[1] < self.width):
            raise NavigationError("Goal position out of bounds")
        
        if np.isinf(self.cost_map[start]):
            raise NavigationError("Start position is impassable")
        
        if np.isinf(self.cost_map[goal]):
            raise NavigationError("Goal position is impassable")
        
        # Initialize
        open_set = []
        heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        nodes_explored = 0
        
        while open_set:
            current = heappop(open_set)[1]
            nodes_explored += 1
            
            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                
                logger.info(
                    "Path found",
                    path_length=len(path),
                    nodes_explored=nodes_explored,
                    total_cost=g_score[goal]
                )
                
                return path
            
            # Check all neighbors
            for neighbor_pos, _, move_cost in self.get_neighbors(current):
                tentative_g = g_score[current] + move_cost
                
                if neighbor_pos not in g_score or tentative_g < g_score[neighbor_pos]:
                    came_from[neighbor_pos] = current
                    g_score[neighbor_pos] = tentative_g
                    f_score[neighbor_pos] = tentative_g + self.heuristic(neighbor_pos, goal)
                    heappush(open_set, (f_score[neighbor_pos], neighbor_pos))
        
        logger.warning("No path found", nodes_explored=nodes_explored)
        return None
    
    def find_path_with_waypoints(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        max_waypoint_spacing: int = 50
    ) -> List[Tuple[int, int]]:
        """Find path and downsample to waypoints.
        
        Args:
            start: Start position
            goal: Goal position
            max_waypoint_spacing: Maximum spacing between waypoints (cells)
        
        Returns:
            List of waypoint positions
        
        Raises:
            NavigationError: If no path exists
        """
        path = self.find_path(start, goal)
        
        if path is None:
            raise NavigationError("No path found between start and goal")
        
        # Downsample path to waypoints
        waypoints = [path[0]]  # Always include start
        
        for i in range(max_waypoint_spacing, len(path), max_waypoint_spacing):
            waypoints.append(path[i])
        
        # Always include goal
        if waypoints[-1] != path[-1]:
            waypoints.append(path[-1])
        
        logger.info(
            "Generated waypoints",
            path_length=len(path),
            num_waypoints=len(waypoints)
        )
        
        return waypoints

