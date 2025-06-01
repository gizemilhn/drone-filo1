from typing import List, Tuple, Dict, Set
import numpy as np
from heapq import heappush, heappop
from datetime import datetime
from zone import NoFlyZone
from drone import Drone
from shapely.geometry import LineString, Polygon

class AStarRouter:
    def __init__(self, grid_size: Tuple[int, int], resolution: float = 1.0):
        self.grid_size = grid_size
        self.resolution = resolution
        self.grid = np.zeros(grid_size)
    
    def _euclidean_distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions."""
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.grid_size[0] and 0 <= new_y < self.grid_size[1]:
                neighbors.append((new_x, new_y))
        return neighbors
    
    def _is_valid_move(self, pos: Tuple[int, int], no_fly_zones: List[NoFlyZone], 
                      current_time: datetime, prev_pos: Tuple[int, int] = None) -> bool:
        """Check if a move is valid considering no-fly zones. Now checks for line crossing."""
        real_pos = (pos[0] * self.resolution, pos[1] * self.resolution)
        if prev_pos is not None:
            prev_real_pos = (prev_pos[0] * self.resolution, prev_pos[1] * self.resolution)
            move_line = LineString([prev_real_pos, real_pos])
            for zone in no_fly_zones:
                if zone.is_active(current_time):
                    poly = Polygon(zone.polygon_coordinates)
                    if move_line.intersects(poly):
                        return False
        else:
            for zone in no_fly_zones:
                if zone.is_active(current_time) and zone.contains_point(real_pos):
                    return False
        return True
    
    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int], 
                  no_fly_zones: List[NoFlyZone], current_time: datetime) -> float:
        """Calculate heuristic cost considering no-fly zones."""
        base_cost = self._euclidean_distance(a, b)
        penalty = 0.0
        
        # Add penalty for proximity to no-fly zones
        real_pos = (a[0] * self.resolution, a[1] * self.resolution)
        for zone in no_fly_zones:
            if zone.is_active(current_time):
                distance = zone.distance_to_boundary(real_pos)
                if distance < 5.0:  # Penalty threshold
                    penalty += (5.0 - distance) * 2.0
        
        return base_cost + penalty
    
    def find_path(self, start: Tuple[float, float], goal: Tuple[float, float],
                 drone: Drone, no_fly_zones: List[NoFlyZone], 
                 current_time: datetime) -> List[Tuple[float, float]]:
        """Find optimal path using A* algorithm."""
        # Convert real coordinates to grid coordinates
        start_grid = (int(start[0] / self.resolution), int(start[1] / self.resolution))
        goal_grid = (int(goal[0] / self.resolution), int(goal[1] / self.resolution))
        
        # Initialize data structures
        open_set: List[Tuple[float, int, Tuple[int, int]]] = []
        closed_set: Set[Tuple[int, int]] = set()
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start_grid: 0}
        f_score: Dict[Tuple[int, int], float] = {start_grid: self._heuristic(start_grid, goal_grid, 
                                                                            no_fly_zones, current_time)}
        
        # Add start node to open set
        heappush(open_set, (f_score[start_grid], 0, start_grid))
        counter = 1  # For tie-breaking in heap
        
        while open_set:
            current_f, _, current = heappop(open_set)
            
            if current == goal_grid:
                # Reconstruct path
                path = []
                while current in came_from:
                    real_pos = (current[0] * self.resolution, current[1] * self.resolution)
                    path.append(real_pos)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            closed_set.add(current)
            
            for neighbor in self._get_neighbors(current):
                if neighbor in closed_set or not self._is_valid_move(neighbor, no_fly_zones, current_time, prev_pos=current):
                    continue
                
                tentative_g = g_score[current] + self._euclidean_distance(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_grid, 
                                                                    no_fly_zones, current_time)
                    heappush(open_set, (f_score[neighbor], counter, neighbor))
                    counter += 1
        
        return []  # No path found
    
    def update_grid(self, no_fly_zones: List[NoFlyZone], current_time: datetime):
        """Update grid with current no-fly zones."""
        self.grid.fill(0)
        for zone in no_fly_zones:
            if zone.is_active(current_time):
                min_coord, max_coord = zone.get_bounding_box()
                min_x = int(min_coord[0] / self.resolution)
                min_y = int(min_coord[1] / self.resolution)
                max_x = int(max_coord[0] / self.resolution)
                max_y = int(max_coord[1] / self.resolution)
                
                for x in range(min_x, max_x + 1):
                    for y in range(min_y, max_y + 1):
                        if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                            real_pos = (x * self.resolution, y * self.resolution)
                            if zone.contains_point(real_pos):
                                self.grid[x, y] = 1 

    def on_optimization_finished(self, assignment):
        try:
            self.system.execute_deliveries(assignment)
            self.update_visualization()
        finally:
            if self.optimizing_dialog:
                self.optimizing_dialog.close()
                self.optimizing_dialog = None
            self.optimizer_thread = None 