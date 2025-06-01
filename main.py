import json
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import numpy as np
from drone import Drone
from delivery import Delivery
from zone import NoFlyZone
from routing import AStarRouter
from optimizer import DeliveryOptimizer, GeneticOptimizer
from visualizer import DeliveryVisualizer

class DroneDeliverySystem:
    def __init__(self, config_file: str = None):
        self.drones: List[Drone] = []
        self.deliveries: List[Delivery] = []
        self.no_fly_zones: List[NoFlyZone] = []
        self.current_time = datetime.now()
        self.grid_size = (100, 100)  # Default grid size
        
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file: str):
        """Load configuration from JSON file."""
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Load drones
        for drone_data in config.get('drones', []):
            self.drones.append(Drone.from_dict(drone_data))
        
        # Load deliveries
        for delivery_data in config.get('deliveries', []):
            self.deliveries.append(Delivery.from_dict(delivery_data))
        
        # Load no-fly zones
        for zone_data in config.get('no_fly_zones', []):
            self.no_fly_zones.append(NoFlyZone.from_dict(zone_data))
        
        # Load grid size if specified
        if 'grid_size' in config:
            self.grid_size = tuple(config['grid_size'])
    
    def save_config(self, config_file: str):
        """Save current configuration to JSON file."""
        config = {
            'drones': [drone.to_dict() for drone in self.drones],
            'deliveries': [delivery.to_dict() for delivery in self.deliveries],
            'no_fly_zones': [zone.to_dict() for zone in self.no_fly_zones],
            'grid_size': self.grid_size
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def add_drone(self, drone: Drone):
        """Add a new drone to the fleet."""
        self.drones.append(drone)
    
    def add_delivery(self, delivery: Delivery):
        """Add a new delivery point."""
        self.deliveries.append(delivery)
    
    def add_no_fly_zone(self, zone: NoFlyZone):
        """Add a new no-fly zone."""
        self.no_fly_zones.append(zone)
    
    def optimize_deliveries(self, use_genetic: bool = False, use_greedy: bool = False) -> Dict:
        """Optimize delivery assignments using either CSP, GA, or Greedy."""
        if use_greedy:
            optimizer = GeneticOptimizer(self.drones, self.deliveries, self.no_fly_zones, self.current_time)
            assignment = optimizer.solve_greedy()
        elif use_genetic:
            optimizer = GeneticOptimizer(self.drones, self.deliveries, self.no_fly_zones, self.current_time)
            try:
                assignment = optimizer.solve()
            except Exception as e:
                print(f"[GA] Exception: {e}. Falling back to greedy.")
                assignment = optimizer.solve_greedy()
        else:
            optimizer = DeliveryOptimizer(self.drones, self.deliveries, self.no_fly_zones, self.current_time)
            assignment = optimizer.solve_csp()
        return assignment
    
    def execute_deliveries(self, assignment: Dict[str, List[Delivery]]):
        """Execute the delivery assignments."""
        router = AStarRouter(self.grid_size)
        for drone_id, deliveries in assignment.items():
            drone = next(d for d in self.drones if d.id == drone_id)
            for delivery in deliveries:
                # Find path to delivery point
                path = router.find_path(drone.current_position, delivery.position,
                                      drone, self.no_fly_zones, self.current_time)
                # Check for no-fly zone intersections
                valid = path and not any(
                    zone.is_active(self.current_time) and any(
                        zone.intersects_line(path[i], path[i+1])
                        for i in range(len(path)-1)
                    ) for zone in self.no_fly_zones)
                if valid:
                    # Update drone position and battery
                    for i in range(len(path) - 1):
                        distance = np.sqrt((path[i+1][0] - path[i][0])**2 +
                                         (path[i+1][1] - path[i][1])**2)
                        drone.update_position(path[i+1], distance)
                    # Mark delivery as completed
                    delivery.mark_completed()
                else:
                    delivery.mark_failed()
    
    def generate_report(self) -> Dict:
        """Generate delivery execution report."""
        report = {
            'total_deliveries': len(self.deliveries),
            'completed_deliveries': sum(1 for d in self.deliveries if d.status == "completed"),
            'failed_deliveries': sum(1 for d in self.deliveries if d.status == "failed"),
            'in_progress_deliveries': sum(1 for d in self.deliveries if d.status == "in_progress"),
            'drone_statistics': {}
        }
        
        for drone in self.drones:
            report['drone_statistics'][drone.id] = {
                'battery_remaining': drone.get_remaining_battery_percentage(),
                'distance_traveled': sum(np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                                      for p1, p2 in zip(drone.route[:-1], drone.route[1:])),
                'deliveries_completed': sum(1 for d in self.deliveries 
                                         if d.assigned_drone == drone.id and d.status == "completed")
            }
        
        return report

def main():
    parser = argparse.ArgumentParser(description='Drone Delivery Fleet Optimization System')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--genetic', action='store_true', help='Use genetic algorithm instead of CSP')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    args = parser.parse_args()
    
    # Initialize system
    system = DroneDeliverySystem(args.config)
    
    # Create visualizer
    visualizer = DeliveryVisualizer(system.grid_size)
    
    # Plot initial state
    if args.visualize:
        visualizer.plot_scenario(system.drones, system.deliveries,
                               system.no_fly_zones, system.current_time)
        visualizer.show()
    
    # Optimize deliveries
    assignment = system.optimize_deliveries(use_genetic=args.genetic)
    
    # Execute deliveries
    system.execute_deliveries(assignment)
    
    # Generate and print report
    report = system.generate_report()
    print("\nDelivery Execution Report:")
    print(f"Total Deliveries: {report['total_deliveries']}")
    print(f"Completed: {report['completed_deliveries']}")
    print(f"Failed: {report['failed_deliveries']}")
    print(f"In Progress: {report['in_progress_deliveries']}")
    
    print("\nDrone Statistics:")
    for drone_id, stats in report['drone_statistics'].items():
        print(f"\nDrone {drone_id}:")
        print(f"  Battery Remaining: {stats['battery_remaining']:.1f}%")
        print(f"  Distance Traveled: {stats['distance_traveled']:.1f} units")
        print(f"  Deliveries Completed: {stats['deliveries_completed']}")
    
    # Plot final state and statistics
    if args.visualize:
        visualizer.plot_scenario(system.drones, system.deliveries,
                               system.no_fly_zones, system.current_time)
        visualizer.plot_statistics(system.drones, system.deliveries)
        visualizer.show()

if __name__ == "__main__":
    main() 