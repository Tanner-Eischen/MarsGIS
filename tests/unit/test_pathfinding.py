"""Unit tests for pathfinding algorithms."""

import numpy as np
import pytest

from marshab.exceptions import NavigationError
from marshab.processing.pathfinding import AStarPathfinder, smooth_path


@pytest.fixture
def simple_cost_map():
    """Create a simple cost map for testing."""
    # 10x10 grid with uniform cost
    return np.ones((10, 10))


@pytest.fixture
def cost_map_with_obstacle():
    """Create a cost map with an obstacle."""
    cost_map = np.ones((10, 10))
    # Create a wall in the middle (rows 4-5, columns 3-7 to leave paths around)
    cost_map[4:6, 3:7] = np.inf
    return cost_map


@pytest.fixture
def cost_map_fully_blocked():
    """Create a cost map with no path from (0,0) to (9,9)."""
    cost_map = np.ones((10, 10))
    # Block everything except start and goal positions
    # Make start (0,0) and goal (9,9) passable, but block all paths between them
    # Strategy: Block all cells except start and goal, ensuring no path exists
    cost_map[:, :] = np.inf  # Block everything first
    cost_map[0, 0] = 1.0  # Start is passable
    cost_map[9, 9] = 1.0  # Goal is passable
    # Block all neighbors of start and goal to ensure no path exists
    # Block all neighbors of (0,0)
    cost_map[1, 0] = np.inf  # Down from start
    cost_map[0, 1] = np.inf  # Right from start
    cost_map[1, 1] = np.inf  # Diagonal from start
    # Block all neighbors of (9,9)
    cost_map[8, 9] = np.inf  # Up from goal
    cost_map[9, 8] = np.inf  # Left from goal
    cost_map[8, 8] = np.inf  # Diagonal from goal
    return cost_map


def test_astar_simple_path(simple_cost_map):
    """Test A* finds a simple straight-line path."""
    pathfinder = AStarPathfinder(simple_cost_map, cell_size_m=1.0)
    
    start = (0, 0)
    goal = (9, 9)
    
    path = pathfinder.find_path(start, goal)
    
    assert path is not None
    assert len(path) > 0
    assert path[0] == start
    assert path[-1] == goal
    
    # Path should be monotonic (no backtracking in simple case)
    # Check that we're generally moving toward goal
    for i in range(len(path) - 1):
        dist1 = np.sqrt((path[i][0] - goal[0])**2 + (path[i][1] - goal[1])**2)
        dist2 = np.sqrt((path[i+1][0] - goal[0])**2 + (path[i+1][1] - goal[1])**2)
        # Distance should generally decrease (allow some tolerance for optimal path)
        assert dist2 <= dist1 + 1.0


def test_astar_path_with_obstacle(cost_map_with_obstacle):
    """Test A* finds path around obstacle."""
    pathfinder = AStarPathfinder(cost_map_with_obstacle, cell_size_m=1.0)
    
    start = (0, 5)
    goal = (9, 5)
    
    path = pathfinder.find_path(start, goal)
    
    assert path is not None
    assert path[0] == start
    assert path[-1] == goal
    
    # Path should avoid the obstacle (rows 4-5, columns 3-6 are blocked)
    # Check that path doesn't go through the blocked area
    for pos in path:
        row, col = pos
        if row in [4, 5] and col in [3, 4, 5, 6]:
            pytest.fail(f"Path goes through blocked area at position {pos}")


def test_astar_no_path(cost_map_fully_blocked):
    """Test A* returns None when no path exists."""
    pathfinder = AStarPathfinder(cost_map_fully_blocked, cell_size_m=1.0)
    
    start = (0, 0)
    goal = (9, 9)
    
    path = pathfinder.find_path(start, goal)
    
    assert path is None


def test_astar_start_out_of_bounds(simple_cost_map):
    """Test A* raises error for out-of-bounds start."""
    pathfinder = AStarPathfinder(simple_cost_map, cell_size_m=1.0)
    
    with pytest.raises(NavigationError, match="out of bounds"):
        pathfinder.find_path((-1, 0), (5, 5))


def test_astar_goal_out_of_bounds(simple_cost_map):
    """Test A* raises error for out-of-bounds goal."""
    pathfinder = AStarPathfinder(simple_cost_map, cell_size_m=1.0)
    
    with pytest.raises(NavigationError, match="out of bounds"):
        pathfinder.find_path((0, 0), (10, 10))


def test_astar_start_impassable(simple_cost_map):
    """Test A* raises error when start is impassable."""
    cost_map = simple_cost_map.copy()
    cost_map[0, 0] = np.inf
    
    pathfinder = AStarPathfinder(cost_map, cell_size_m=1.0)
    
    with pytest.raises(NavigationError, match="impassable"):
        pathfinder.find_path((0, 0), (5, 5))


def test_astar_goal_impassable(simple_cost_map):
    """Test A* raises error when goal is impassable."""
    cost_map = simple_cost_map.copy()
    cost_map[9, 9] = np.inf
    
    pathfinder = AStarPathfinder(cost_map, cell_size_m=1.0)
    
    with pytest.raises(NavigationError, match="impassable"):
        pathfinder.find_path((0, 0), (9, 9))


def test_astar_heuristic():
    """Test heuristic function."""
    cost_map = np.ones((10, 10))
    pathfinder = AStarPathfinder(cost_map, cell_size_m=1.0)
    
    # Heuristic should be Euclidean distance
    h = pathfinder.heuristic((0, 0), (3, 4))
    expected = np.sqrt(3**2 + 4**2) * 1.0  # 5.0
    assert abs(h - expected) < 0.01


def test_astar_get_neighbors(simple_cost_map):
    """Test neighbor generation."""
    pathfinder = AStarPathfinder(simple_cost_map, cell_size_m=1.0)
    
    # Center position should have 8 neighbors
    neighbors = pathfinder.get_neighbors((5, 5))
    assert len(neighbors) == 8
    
    # Corner position should have 3 neighbors
    neighbors = pathfinder.get_neighbors((0, 0))
    assert len(neighbors) == 3
    
    # Edge position should have 5 neighbors
    neighbors = pathfinder.get_neighbors((0, 5))
    assert len(neighbors) == 5


def test_astar_get_neighbors_obstacles(cost_map_with_obstacle):
    """Test neighbor generation excludes obstacles."""
    pathfinder = AStarPathfinder(cost_map_with_obstacle, cell_size_m=1.0)
    
    # Position next to obstacle should have fewer neighbors
    neighbors = pathfinder.get_neighbors((3, 5))
    # Should not include positions in rows 4-5 (obstacle)
    # get_neighbors returns (row, col, cost) tuples
    for row, col, cost in neighbors:
        assert row not in [4, 5]


def test_astar_waypoints_simple(simple_cost_map):
    """Test waypoint generation."""
    pathfinder = AStarPathfinder(simple_cost_map, cell_size_m=1.0)
    
    start = (0, 0)
    goal = (9, 9)
    
    waypoints = pathfinder.find_path_with_waypoints(start, goal, max_waypoint_spacing=3)
    
    assert len(waypoints) > 0
    assert waypoints[0] == start
    assert waypoints[-1] == goal
    
    # Waypoints should be subset of full path
    full_path = pathfinder.find_path(start, goal)
    assert full_path is not None
    assert all(wp in full_path for wp in waypoints)


def test_astar_waypoints_spacing(simple_cost_map):
    """Test waypoint spacing."""
    pathfinder = AStarPathfinder(simple_cost_map, cell_size_m=1.0)
    
    start = (0, 0)
    goal = (20, 20)
    cost_map = np.ones((25, 25))
    pathfinder = AStarPathfinder(cost_map, cell_size_m=1.0)
    
    max_waypoint_spacing = 5  # Spacing parameter
    waypoints = pathfinder.find_path_with_waypoints(start, goal, max_waypoint_spacing=max_waypoint_spacing)
    
    # Check spacing between consecutive waypoints (except last)
    for i in range(len(waypoints) - 1):
        dist = np.sqrt(
            (waypoints[i+1][0] - waypoints[i][0])**2 +
            (waypoints[i+1][1] - waypoints[i][1])**2
        )
        # Should be approximately max_waypoint_spacing (allow some tolerance)
        assert dist <= max_waypoint_spacing * 1.5


def test_astar_waypoints_no_path(cost_map_fully_blocked):
    """Test waypoint generation raises error when no path exists."""
    pathfinder = AStarPathfinder(cost_map_fully_blocked, cell_size_m=1.0)
    
    with pytest.raises(NavigationError, match="No path found"):
        pathfinder.find_path_with_waypoints((0, 0), (9, 9))


def test_astar_diagonal_cost(simple_cost_map):
    """Test that diagonal moves have higher cost."""
    pathfinder = AStarPathfinder(simple_cost_map, cell_size_m=1.0)
    
    neighbors = pathfinder.get_neighbors((5, 5))
    
    # Find diagonal and cardinal neighbors
    diagonal_costs = []
    cardinal_costs = []
    
    for row, col, cost in neighbors:
        dr = abs(row - 5)
        dc = abs(col - 5)
        if dr == 1 and dc == 1:  # Diagonal
            diagonal_costs.append(cost)
        elif (dr == 1 and dc == 0) or (dr == 0 and dc == 1):  # Cardinal
            cardinal_costs.append(cost)
    
    # Diagonal should cost more (1.414 vs 1.0)
    if diagonal_costs and cardinal_costs:
        assert min(diagonal_costs) > max(cardinal_costs)


def test_astar_optimal_path():
    """Test that A* finds optimal path in simple scenario."""
    # Create a cost map where there's a clear optimal path
    cost_map = np.ones((10, 10))
    # Make a "highway" with lower cost
    cost_map[:, 5] = 0.5
    
    pathfinder = AStarPathfinder(cost_map, cell_size_m=1.0)
    
    start = (0, 0)
    goal = (9, 9)
    
    path = pathfinder.find_path(start, goal)
    
    assert path is not None
    # Optimal path should use the highway (column 5)
    # Check if any point in path uses the highway
    uses_highway = any(pos[1] == 5 for pos in path)
    assert uses_highway


def test_smooth_path():
    """Test path smoothing."""
    # Create a simple cost map
    cost_map = np.ones((20, 20))
    
    # Create a path with unnecessary waypoints
    path = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (10, 10), (15, 15), (19, 19)]
    
    smoothed = smooth_path(path, cost_map, tolerance=2.0)
    
    # Smoothed path should be shorter or equal
    assert len(smoothed) <= len(path)
    # Should still start and end at same points
    assert smoothed[0] == path[0]
    assert smoothed[-1] == path[-1]


def test_smooth_path_with_obstacle():
    """Test path smoothing respects obstacles."""
    # Create cost map with obstacle
    cost_map = np.ones((20, 20))
    cost_map[5:15, 5:15] = np.inf  # Obstacle in middle
    
    # Path that goes around obstacle
    path = [(0, 0), (5, 0), (10, 0), (15, 0), (19, 0), (19, 10), (19, 19)]
    
    smoothed = smooth_path(path, cost_map, tolerance=2.0)
    
    # Should not smooth through obstacle
    # Check that no waypoint is in the obstacle
    for waypoint in smoothed:
        row, col = waypoint
        assert not (5 <= row < 15 and 5 <= col < 15)


def test_smooth_path_short_path():
    """Test path smoothing with very short path."""
    cost_map = np.ones((10, 10))
    path = [(0, 0), (9, 9)]
    
    smoothed = smooth_path(path, cost_map, tolerance=2.0)
    
    # Short paths should not be modified
    assert len(smoothed) == len(path)
    assert smoothed == path


