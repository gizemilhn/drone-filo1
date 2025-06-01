from dataclasses import dataclass
from typing import List, Tuple
from datetime import datetime
from shapely.geometry import Polygon, Point
import numpy as np

@dataclass
class NoFlyZone:
    id: str
    polygon_coordinates: List[Tuple[float, float]]
    active_time_start: datetime
    active_time_end: datetime
    
    def __post_init__(self):
        """Initialize the polygon and validate coordinates."""
        self.polygon = Polygon(self.polygon_coordinates)
        if not self.polygon.is_valid:
            raise ValueError("Invalid polygon coordinates")
    
    def is_active(self, current_time: datetime) -> bool:
        """Check if the no-fly zone is active at the given time."""
        return self.active_time_start <= current_time <= self.active_time_end
    
    def contains_point(self, point: Tuple[float, float]) -> bool:
        """Check if a point is inside the no-fly zone."""
        return self.polygon.contains(Point(point))
    
    def distance_to_boundary(self, point: Tuple[float, float]) -> float:
        """Calculate minimum distance from point to zone boundary."""
        point_obj = Point(point)
        return point_obj.distance(self.polygon.boundary)
    
    def intersects_line(self, start: Tuple[float, float], end: Tuple[float, float]) -> bool:
        """Check if a line segment intersects with the no-fly zone."""
        from shapely.geometry import LineString
        line = LineString([start, end])
        return self.polygon.intersects(line)
    
    def get_centroid(self) -> Tuple[float, float]:
        """Get the centroid of the no-fly zone."""
        centroid = self.polygon.centroid
        return (centroid.x, centroid.y)
    
    def to_dict(self) -> dict:
        """Convert no-fly zone to dictionary for serialization."""
        return {
            'id': self.id,
            'polygon_coordinates': self.polygon_coordinates,
            'active_time_start': self.active_time_start.isoformat(),
            'active_time_end': self.active_time_end.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'NoFlyZone':
        """Create no-fly zone instance from dictionary."""
        # Convert ISO format strings back to datetime objects
        data['active_time_start'] = datetime.fromisoformat(data['active_time_start'])
        data['active_time_end'] = datetime.fromisoformat(data['active_time_end'])
        return cls(**data)
    
    def get_bounding_box(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get the bounding box of the no-fly zone."""
        minx, miny, maxx, maxy = self.polygon.bounds
        return ((minx, miny), (maxx, maxy)) 