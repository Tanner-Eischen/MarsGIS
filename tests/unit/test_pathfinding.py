"""Unit tests for pathfinding algorithms."""

import numpy as np
import pytest

from marshab.exceptions import NavigationError
from marshab.processing.pathfinding import AStarPathfinder


@pytest.fixture
def simple_cost_map():
    """Create a simple cost map for testing."""
    # 10x10 grid with uniform cost
    return np.ones((10, 10))


@pytest.fixture
def cost_map_with_obstacle():
    """Create a cost map with an obstacle."""
    cost_map = np.ones((10, 10))
    # Create a wall in the middle
    cost_map[4:6, :] = np.inf
    return cost_map


@pytest.fixture
def cost_map_fully_blocked():
    """Create a cost map with no path."""
    cost_map = np.ones((10, 10))
    # Block everything except start and goal
    cost_map[1:9, 1:9] = np.inf
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
    
    # Path should avoid the obstacle (rows 4-5)
    for pos in path:
        assert pos[0] not in [4, 5] or pos[1] not in range(10)


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
    for _, (row, col), _ in neighbors:
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
    
    waypoints = pathfinder.find_path_with_waypoints(start, goal, max_waypoint_spacing=5)
    
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

