from dataclasses import dataclass
from typing import Tuple, List
import numpy as np

@dataclass
class Drone:
    id: str
    max_weight: float
    battery_capacity: float
    speed: float
    start_position: Tuple[float, float]
    current_position: Tuple[float, float] = None
    current_battery: float = None
    current_weight: float = 0.0
    route: List[Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.current_position is None:
            self.current_position = self.start_position
        if self.current_battery is None:
            self.current_battery = self.battery_capacity
        if self.route is None:
            self.route = [self.start_position]
    
    def can_carry(self, weight: float) -> bool:
        """Check if drone can carry additional weight."""
        return self.current_weight + weight <= self.max_weight
    
    def has_sufficient_battery(self, distance: float) -> bool:
        """Check if drone has enough battery for given distance."""
        energy_needed = distance / self.speed
        return self.current_battery >= energy_needed
    
    def update_position(self, new_position: Tuple[float, float], distance: float):
        """Update drone position and battery level."""
        self.current_position = new_position
        self.current_battery -= distance / self.speed
        self.route.append(new_position)
    
    def reset(self):
        """Reset drone to initial state."""
        self.current_position = self.start_position
        self.current_battery = self.battery_capacity
        self.current_weight = 0.0
        self.route = [self.start_position]
    
    def get_remaining_battery_percentage(self) -> float:
        """Get remaining battery as percentage."""
        return (self.current_battery / self.battery_capacity) * 100
    
    def to_dict(self) -> dict:
        """Convert drone to dictionary for serialization."""
        return {
            'id': self.id,
            'max_weight': self.max_weight,
            'battery_capacity': self.battery_capacity,
            'speed': self.speed,
            'start_position': self.start_position,
            'current_position': self.current_position,
            'current_battery': self.current_battery,
            'current_weight': self.current_weight,
            'route': self.route
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Drone':
        """Create drone instance from dictionary."""
        return cls(**data) 