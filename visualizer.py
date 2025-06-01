import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime
from drone import Drone
from delivery import Delivery
from zone import NoFlyZone

class DeliveryVisualizer:
    def __init__(self, grid_size: Tuple[float, float]):
        self.grid_size = grid_size
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.colors = plt.cm.rainbow(np.linspace(0, 1, 10))  # For different drones
    
    def plot_scenario(self, drones: List[Drone], deliveries: List[Delivery],
                     no_fly_zones: List[NoFlyZone], current_time: datetime):
        """Plot the complete delivery scenario."""
        self.ax.clear()
        self.ax.set_xlim(0, self.grid_size[0])
        self.ax.set_ylim(0, self.grid_size[1])
        self.ax.set_title("Drone Delivery Fleet Visualization")
        self.ax.set_xlabel("X Coordinate")
        self.ax.set_ylabel("Y Coordinate")
        
        # Plot no-fly zones
        self._plot_no_fly_zones(no_fly_zones, current_time)
        
        # Plot delivery points
        self._plot_delivery_points(deliveries)
        
        # Plot drones and their routes
        self._plot_drones_and_routes(drones)
        
        # Add legend
        self.ax.legend()
        
        plt.tight_layout()
    
    def _plot_no_fly_zones(self, no_fly_zones: List[NoFlyZone], current_time: datetime):
        """Plot active no-fly zones."""
        for zone in no_fly_zones:
            if zone.is_active(current_time):
                polygon = patches.Polygon(zone.polygon_coordinates, 
                                       facecolor='red', alpha=0.3,
                                       label='No-Fly Zone')
                self.ax.add_patch(polygon)
    
    def _plot_delivery_points(self, deliveries: List[Delivery]):
        """Plot delivery points with different markers based on status."""
        for delivery in deliveries:
            color = 'green' if delivery.status == "completed" else \
                   'red' if delivery.status == "failed" else \
                   'blue' if delivery.status == "in_progress" else 'gray'
            
            self.ax.scatter(delivery.position[0], delivery.position[1],
                          c=color, marker='o', s=100,
                          label=f'Delivery {delivery.id} ({delivery.status})')
            
            # Add priority label
            self.ax.text(delivery.position[0], delivery.position[1] + 0.5,
                        f'P{delivery.priority}', ha='center')
    
    def _plot_drones_and_routes(self, drones: List[Drone]):
        """Plot drones and their routes."""
        for i, drone in enumerate(drones):
            color = self.colors[i % len(self.colors)]
            
            # Plot drone's current position
            self.ax.scatter(drone.current_position[0], drone.current_position[1],
                          c=[color], marker='^', s=150,
                          label=f'Drone {drone.id}')
            
            # Plot drone's route
            if len(drone.route) > 1:
                route_x = [p[0] for p in drone.route]
                route_y = [p[1] for p in drone.route]
                self.ax.plot(route_x, route_y, c=color, alpha=0.5, linestyle='--')
            
            # Add battery level
            battery_text = f'Battery: {drone.get_remaining_battery_percentage():.1f}%'
            self.ax.text(drone.current_position[0], drone.current_position[1] - 0.5,
                        battery_text, ha='center', fontsize=8)
    
    def show(self):
        """Display the plot."""
        plt.show()
    
    def save(self, filename: str):
        """Save the plot to a file."""
        plt.savefig(filename)
    
    def plot_statistics(self, drones: List[Drone], deliveries: List[Delivery]):
        """Plot delivery statistics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot delivery status distribution
        status_counts = {'completed': 0, 'failed': 0, 'in_progress': 0, 'pending': 0}
        for delivery in deliveries:
            status_counts[delivery.status] += 1
        
        ax1.bar(status_counts.keys(), status_counts.values())
        ax1.set_title('Delivery Status Distribution')
        ax1.set_ylabel('Number of Deliveries')
        
        # Plot drone battery levels
        drone_ids = [drone.id for drone in drones]
        battery_levels = [drone.get_remaining_battery_percentage() for drone in drones]
        
        ax2.bar(drone_ids, battery_levels)
        ax2.set_title('Drone Battery Levels')
        ax2.set_ylabel('Battery Level (%)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_optimization_comparison(self, csp_results: Dict, ga_results: Dict):
        """Plot comparison between CSP and GA optimization results."""
        metrics = ['completed_deliveries', 'total_distance', 'energy_usage']
        csp_values = [csp_results[m] for m in metrics]
        ga_values = [ga_results[m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, csp_values, width, label='CSP')
        ax.bar(x + width/2, ga_values, width, label='GA')
        
        ax.set_ylabel('Value')
        ax.set_title('Optimization Algorithm Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        plt.tight_layout()
        plt.show() 