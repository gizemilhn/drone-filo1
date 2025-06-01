from dataclasses import dataclass
from typing import Tuple, Optional
from datetime import datetime, timedelta

@dataclass
class Delivery:
    id: str
    position: Tuple[float, float]
    weight: float
    priority: int
    time_window_start: datetime
    time_window_end: datetime
    assigned_drone: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    
    def is_within_time_window(self, current_time: datetime) -> bool:
        """Check if current time is within delivery time window."""
        return self.time_window_start <= current_time <= self.time_window_end
    
    def time_until_deadline(self, current_time: datetime) -> timedelta:
        """Calculate time remaining until deadline."""
        return self.time_window_end - current_time
    
    def assign_to_drone(self, drone_id: str):
        """Assign delivery to a drone."""
        self.assigned_drone = drone_id
        self.status = "in_progress"
    
    def mark_completed(self):
        """Mark delivery as completed."""
        self.status = "completed"
    
    def mark_failed(self):
        """Mark delivery as failed."""
        self.status = "failed"
    
    def to_dict(self) -> dict:
        """Convert delivery to dictionary for serialization."""
        return {
            'id': self.id,
            'position': self.position,
            'weight': self.weight,
            'priority': self.priority,
            'time_window_start': self.time_window_start.isoformat(),
            'time_window_end': self.time_window_end.isoformat(),
            'assigned_drone': self.assigned_drone,
            'status': self.status
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Delivery':
        """Create delivery instance from dictionary."""
        # Convert ISO format strings back to datetime objects
        data['time_window_start'] = datetime.fromisoformat(data['time_window_start'])
        data['time_window_end'] = datetime.fromisoformat(data['time_window_end'])
        return cls(**data)
    
    def __lt__(self, other: 'Delivery') -> bool:
        """Compare deliveries based on priority and deadline."""
        if self.priority != other.priority:
            return self.priority > other.priority  # Higher priority first
        return self.time_window_end < other.time_window_end  # Earlier deadline first 