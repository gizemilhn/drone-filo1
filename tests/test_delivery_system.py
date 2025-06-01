import unittest
from datetime import datetime, timedelta
from drone import Drone
from delivery import Delivery
from zone import NoFlyZone
from routing import AStarRouter
from optimizer import DeliveryOptimizer, GeneticOptimizer
from main import DroneDeliverySystem

class TestDroneDeliverySystem(unittest.TestCase):
    def setUp(self):
        # Create test drones
        self.drones = [
            Drone("drone1", 5.0, 100.0, 10.0, (0.0, 0.0)),
            Drone("drone2", 7.0, 120.0, 12.0, (100.0, 0.0))
        ]
        
        # Create test deliveries
        current_time = datetime.now()
        self.deliveries = [
            Delivery("delivery1", (20.0, 30.0), 2.0, 1,
                    current_time, current_time + timedelta(hours=1)),
            Delivery("delivery2", (70.0, 40.0), 3.0, 2,
                    current_time, current_time + timedelta(hours=1))
        ]
        
        # Create test no-fly zones
        self.no_fly_zones = [
            NoFlyZone("zone1", [(30.0, 30.0), (40.0, 30.0),
                               (40.0, 40.0), (30.0, 40.0)],
                     current_time, current_time + timedelta(hours=2))
        ]
        
        # Initialize system
        self.system = DroneDeliverySystem()
        for drone in self.drones:
            self.system.add_drone(drone)
        for delivery in self.deliveries:
            self.system.add_delivery(delivery)
        for zone in self.no_fly_zones:
            self.system.add_no_fly_zone(zone)
    
    def test_drone_initialization(self):
        """Test drone initialization and properties."""
        drone = self.drones[0]
        self.assertEqual(drone.id, "drone1")
        self.assertEqual(drone.max_weight, 5.0)
        self.assertEqual(drone.battery_capacity, 100.0)
        self.assertEqual(drone.speed, 10.0)
        self.assertEqual(drone.current_position, (0.0, 0.0))
    
    def test_delivery_initialization(self):
        """Test delivery initialization and properties."""
        delivery = self.deliveries[0]
        self.assertEqual(delivery.id, "delivery1")
        self.assertEqual(delivery.position, (20.0, 30.0))
        self.assertEqual(delivery.weight, 2.0)
        self.assertEqual(delivery.priority, 1)
        self.assertEqual(delivery.status, "pending")
    
    def test_no_fly_zone_initialization(self):
        """Test no-fly zone initialization and properties."""
        zone = self.no_fly_zones[0]
        self.assertEqual(zone.id, "zone1")
        self.assertEqual(len(zone.polygon_coordinates), 4)
        self.assertTrue(zone.is_active(datetime.now()))
    
    def test_drone_can_carry(self):
        """Test drone weight capacity check."""
        drone = self.drones[0]
        self.assertTrue(drone.can_carry(2.0))
        self.assertFalse(drone.can_carry(6.0))
    
    def test_delivery_time_window(self):
        """Test delivery time window validation."""
        delivery = self.deliveries[0]
        current_time = datetime.now()
        self.assertTrue(delivery.is_within_time_window(current_time))
        self.assertFalse(delivery.is_within_time_window(current_time + timedelta(hours=2)))
    
    def test_no_fly_zone_contains_point(self):
        """Test no-fly zone point containment."""
        zone = self.no_fly_zones[0]
        self.assertTrue(zone.contains_point((35.0, 35.0)))
        self.assertFalse(zone.contains_point((50.0, 50.0)))
    
    def test_optimization(self):
        """Test delivery optimization."""
        # Test CSP optimization
        csp_optimizer = DeliveryOptimizer(self.drones, self.deliveries,
                                        self.no_fly_zones, datetime.now())
        csp_assignment = csp_optimizer.solve_csp()
        self.assertIsInstance(csp_assignment, dict)
        
        # Test GA optimization
        ga_optimizer = GeneticOptimizer(self.drones, self.deliveries,
                                      self.no_fly_zones, datetime.now())
        ga_assignment = ga_optimizer.solve()
        self.assertIsInstance(ga_assignment, dict)
    
    def test_system_execution(self):
        """Test complete system execution."""
        # Optimize deliveries
        assignment = self.system.optimize_deliveries()
        
        # Execute deliveries
        self.system.execute_deliveries(assignment)
        
        # Generate report
        report = self.system.generate_report()
        self.assertIn('total_deliveries', report)
        self.assertIn('completed_deliveries', report)
        self.assertIn('failed_deliveries', report)
        self.assertIn('drone_statistics', report)

if __name__ == '__main__':
    unittest.main() 