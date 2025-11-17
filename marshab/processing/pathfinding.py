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
        # Ensure a and b are tuples
        if not isinstance(a, (tuple, list)) or len(a) != 2:
            raise ValueError(f"heuristic: a must be (row, col) tuple, got {type(a)}: {a}")
        if not isinstance(b, (tuple, list)) or len(b) != 2:
            raise ValueError(f"heuristic: b must be (row, col) tuple, got {type(b)}: {b}")
        a = tuple(a)
        b = tuple(b)
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
        
        # Validate start and goal are tuples
        if not isinstance(start, (tuple, list)) or len(start) != 2:
            raise NavigationError(f"Start must be a tuple (row, col), got {type(start)}: {start}")
        if not isinstance(goal, (tuple, list)) or len(goal) != 2:
            raise NavigationError(f"Goal must be a tuple (row, col), got {type(goal)}: {goal}")
        
        # Convert to tuple if list
        start = tuple(start)
        goal = tuple(goal)
        
        # Validate start and goal
        if not (0 <= start[0] < self.height and 0 <= start[1] < self.width):
            raise NavigationError("Start position out of bounds")
        
        if not (0 <= goal[0] < self.height and 0 <= goal[1] < self.width):
            raise NavigationError("Goal position out of bounds")
        
        if np.isinf(self.cost_map[start[0], start[1]]):
            raise NavigationError("Start position is impassable")
        
        if np.isinf(self.cost_map[goal[0], goal[1]]):
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
            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                # Ensure neighbor is a tuple of (row, col, cost)
                if not isinstance(neighbor, (tuple, list)) or len(neighbor) != 3:
                    logger.error(f"Invalid neighbor format: {neighbor}, type: {type(neighbor)}")
                    continue
                
                neighbor_pos = (int(neighbor[0]), int(neighbor[1]))
                move_cost = float(neighbor[2])
                
                # Ensure neighbor_pos is a tuple
                if not isinstance(neighbor_pos, tuple) or len(neighbor_pos) != 2:
                    logger.error(f"Invalid neighbor_pos after extraction: {neighbor_pos}, type: {type(neighbor_pos)}")
                    continue
                
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
        
        # Validate path format
        if not isinstance(path, list) or len(path) == 0:
            raise NavigationError(f"Invalid path format: {type(path)}, length: {len(path) if hasattr(path, '__len__') else 'N/A'}")
        
        # Ensure path elements are tuples
        if not all(isinstance(p, (tuple, list)) and len(p) == 2 for p in path):
            raise NavigationError(f"Path contains invalid elements. First element: {path[0] if path else 'N/A'}, type: {type(path[0]) if path else 'N/A'}")
        
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


def smooth_path(
    path: List[Tuple[int, int]],
    cost_map: np.ndarray,
    tolerance: float = 2.0,
) -> List[Tuple[int, int]]:
    """Smooth path by removing unnecessary waypoints using line-of-sight checks.
    
    For each waypoint, checks if a direct path to future waypoints is clear.
    Removes intermediate waypoints if the direct path has no obstacles.
    
    Args:
        path: List of (row, col) positions forming the path
        cost_map: 2D array of traversability costs (inf = impassable)
        tolerance: Maximum deviation from original path (cells)
    
    Returns:
        Smoothed path with fewer waypoints
    """
    if len(path) <= 2:
        # Path too short to smooth
        return path
    
    logger.debug(
        "Smoothing path",
        original_length=len(path),
        tolerance=tolerance
    )
    
    smoothed = [path[0]]  # Always keep start
    
    i = 0
    while i < len(path) - 1:
        # Try to skip ahead as far as possible
        best_j = i + 1
        
        # Check if we can skip directly to goal
        if i == 0 and _has_line_of_sight(path[i], path[-1], cost_map, tolerance):
            # Can go directly to goal, skip everything
            smoothed.append(path[-1])
            break
        
        # Try to skip ahead to future waypoints
        for j in range(len(path) - 1, i, -1):
            if _has_line_of_sight(path[i], path[j], cost_map, tolerance):
                best_j = j
                break
        
        # Add the furthest reachable waypoint
        if best_j > i + 1:
            smoothed.append(path[best_j])
            i = best_j
        else:
            # Can't skip, add next waypoint
            smoothed.append(path[i + 1])
            i += 1
    
    # Always include goal
    if smoothed[-1] != path[-1]:
        smoothed.append(path[-1])
    
    logger.info(
        "Path smoothed",
        original_length=len(path),
        smoothed_length=len(smoothed),
        reduction=len(path) - len(smoothed)
    )
    
    return smoothed


def _has_line_of_sight(
    start: Tuple[int, int],
    end: Tuple[int, int],
    cost_map: np.ndarray,
    tolerance: float = 2.0,
) -> bool:
    """Check if there's a clear line of sight between two points.
    
    Uses Bresenham's line algorithm to check all cells along the path.
    
    Args:
        start: Start position (row, col)
        end: End position (row, col)
        cost_map: 2D array of traversability costs
        tolerance: Maximum deviation from straight line (cells)
    
    Returns:
        True if path is clear (no impassable terrain)
    """
    start_row, start_col = start
    end_row, end_col = end
    
    # Check if start or end is out of bounds
    height, width = cost_map.shape
    if not (0 <= start_row < height and 0 <= start_col < width):
        return False
    if not (0 <= end_row < height and 0 <= end_col < width):
        return False
    
    # Check if start or end is impassable
    if np.isinf(cost_map[start_row, start_col]):
        return False
    if np.isinf(cost_map[end_row, end_col]):
        return False
    
    # Use Bresenham's line algorithm to check all cells along the path
    # Simplified version - check cells along the line
    row_diff = abs(end_row - start_row)
    col_diff = abs(end_col - start_col)
    
    # Determine step direction
    row_step = 1 if end_row > start_row else -1
    col_step = 1 if end_col > start_col else -1
    
    # Check cells along the line
    current_row, current_col = start_row, start_col
    error = 0
    
    # Use the larger dimension to determine the primary direction
    if row_diff > col_diff:
        # Row is primary direction
        for _ in range(row_diff + 1):
            # Check if current cell is impassable
            if 0 <= current_row < height and 0 <= current_col < width:
                if np.isinf(cost_map[current_row, current_col]):
                    return False
            
            # Move to next cell
            error += col_diff
            if error >= row_diff:
                current_col += col_step
                error -= row_diff
            current_row += row_step
    else:
        # Column is primary direction
        for _ in range(col_diff + 1):
            # Check if current cell is impassable
            if 0 <= current_row < height and 0 <= current_col < width:
                if np.isinf(cost_map[current_row, current_col]):
                    return False
            
            # Move to next cell
            error += row_diff
            if error >= col_diff:
                current_row += row_step
                error -= col_diff
            current_col += col_step
    
    return True


