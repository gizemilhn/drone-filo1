from typing import List, Dict, Tuple, Set
import random
import numpy as np
from datetime import datetime, timedelta
from drone import Drone
from delivery import Delivery
from zone import NoFlyZone
from routing import AStarRouter
import time

class DeliveryOptimizer:
    def __init__(self, drones: List[Drone], deliveries: List[Delivery], 
                 no_fly_zones: List[NoFlyZone], current_time: datetime):
        self.drones = drones
        self.deliveries = sorted(deliveries)  # Sort by priority and deadline
        self.no_fly_zones = no_fly_zones
        self.current_time = current_time
        self.assignment: Dict[str, List[Delivery]] = {drone.id: [] for drone in drones}
        self.router = AStarRouter((100, 100))  # Use default grid size or pass as needed
    
    def solve_csp(self) -> Dict[str, List[Delivery]]:
        """Improved CSP: Assign deliveries to drones in sequence, maximizing completed deliveries."""
        # Sort deliveries by priority and earliest time window
        deliveries = sorted(self.deliveries, key=lambda d: (-d.priority, d.time_window_start))
        # Reset drone states
        drone_states = {drone.id: {
            'position': drone.current_position,
            'battery': drone.current_battery,
            'time': self.current_time,
            'route': list(drone.route)
        } for drone in self.drones}
        assignment = {drone.id: [] for drone in self.drones}
        for delivery in deliveries:
            best_drone = None
            best_arrival_time = None
            best_path = None
            for drone in self.drones:
                state = drone_states[drone.id]
                # Check weight
                if delivery.weight > drone.max_weight:
                    continue
                # Plan path from current state
                path = self.router.find_path(state['position'], delivery.position, drone, self.no_fly_zones, state['time'])
                if not path or any(
                    zone.is_active(state['time']) and any(
                        zone.intersects_line(path[i], path[i+1])
                        for i in range(len(path)-1)
                    ) for zone in self.no_fly_zones):
                    continue
                # Calculate arrival time and battery usage
                total_dist = sum(np.sqrt((path[i+1][0] - path[i][0])**2 + (path[i+1][1] - path[i][1])**2) for i in range(len(path)-1))
                travel_time = total_dist / drone.speed
                arrival_time = state['time'] + timedelta(hours=travel_time)
                battery_used = total_dist  # 1 unit per distance (customize as needed)
                if battery_used > state['battery']:
                    continue
                # Check time window
                if not (delivery.time_window_start <= arrival_time <= delivery.time_window_end):
                    continue
                # Choose the soonest possible assignment
                if best_arrival_time is None or arrival_time < best_arrival_time:
                    best_drone = drone
                    best_arrival_time = arrival_time
                    best_path = path
            if best_drone:
                # Assign delivery
                assignment[best_drone.id].append(delivery)
                # Update drone state
                state = drone_states[best_drone.id]
                total_dist = sum(np.sqrt((best_path[i+1][0] - best_path[i][0])**2 + (best_path[i+1][1] - best_path[i][1])**2) for i in range(len(best_path)-1))
                travel_time = total_dist / best_drone.speed
                state['position'] = delivery.position
                state['battery'] -= total_dist
                state['time'] = best_arrival_time
                state['route'].extend(best_path[1:])
                delivery.assigned_drone = best_drone.id
                delivery.status = "completed"
            else:
                delivery.status = "failed"
        # Update drone routes
        for drone in self.drones:
            drone.route = drone_states[drone.id]['route']
            drone.current_position = drone_states[drone.id]['position']
            drone.current_battery = drone_states[drone.id]['battery']
        return assignment
    
    def _is_valid_assignment(self, drone: Drone, delivery: Delivery) -> bool:
        """Check if a delivery can be assigned to a drone."""
        # Check weight capacity
        if not drone.can_carry(delivery.weight):
            return False
        
        # Check time window
        if not delivery.is_within_time_window(self.current_time):
            return False
        
        # Check if drone's current route intersects with any no-fly zones
        for zone in self.no_fly_zones:
            if zone.is_active(self.current_time):
                if any(zone.intersects_line(p1, p2) 
                      for p1, p2 in zip(drone.route[:-1], drone.route[1:])):
                    return False
        
        return True
    
    def _calculate_assignment_score(self, drone: Drone, delivery: Delivery) -> float:
        """Calculate score for a potential assignment."""
        # Distance to delivery point
        distance = np.sqrt((drone.current_position[0] - delivery.position[0])**2 +
                         (drone.current_position[1] - delivery.position[1])**2)
        
        # Time until deadline
        time_until_deadline = delivery.time_until_deadline(self.current_time)
        time_factor = time_until_deadline.total_seconds() / 3600  # Convert to hours
        
        # Battery usage
        battery_usage = distance / drone.speed
        
        return (distance * 0.4 + 
                (1.0 / time_factor) * 0.4 + 
                battery_usage * 0.2)
    
    def _reschedule_deliveries(self, failed_delivery: Delivery) -> bool:
        """Try to reschedule existing deliveries to accommodate failed delivery."""
        for drone_id, deliveries in self.assignment.items():
            for i, delivery in enumerate(deliveries):
                # Try to swap deliveries
                if self._is_valid_assignment(self.drones[i], failed_delivery):
                    self.assignment[drone_id][i] = failed_delivery
                    return True
        return False

class GeneticOptimizer:
    def __init__(self, drones: List[Drone], deliveries: List[Delivery],
                 no_fly_zones: List[NoFlyZone], current_time: datetime,
                 population_size: int = 10, generations: int = 5, max_time: float = 10.0, early_stop_rounds: int = 3):
        self.drones = drones
        self.deliveries = deliveries
        self.no_fly_zones = no_fly_zones
        self.current_time = current_time
        self.population_size = population_size
        self.generations = generations
        self.router = AStarRouter((100, 100))
        self.max_time = max_time
        self.early_stop_rounds = early_stop_rounds
    
    def solve(self) -> Dict[str, List[Delivery]]:
        """Solve delivery assignment using Genetic Algorithm with early stopping and time limit."""
        population = self._initialize_population()
        best_fitness = float('-inf')
        best_solution = None
        no_improve_rounds = 0
        start_time = time.time()
        for generation in range(self.generations):
            if time.time() - start_time > self.max_time:
                print(f"[GA] Stopping early due to time limit at generation {generation}")
                break
            fitness_scores = [self._calculate_fitness(solution) for solution in population]
            gen_best = max(fitness_scores)
            if gen_best > best_fitness:
                best_fitness = gen_best
                best_solution = population[fitness_scores.index(gen_best)]
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1
            if no_improve_rounds >= self.early_stop_rounds:
                print(f"[GA] Early stopping at generation {generation} (no improvement for {self.early_stop_rounds} rounds)")
                break
            parents = self._select_parents(population, fitness_scores)
            new_population = []
            while len(new_population) < self.population_size:
                if random.random() < 0.2:  # 20% chance to inject random individual
                    new_population.append(self._random_individual())
                else:
                    parent1, parent2 = random.sample(parents, 2)
                    child = self._crossover(parent1, parent2)
                    child = self._mutate(child)
                    new_population.append(child)
            population = new_population
        if best_solution is None:
            best_solution = max(population, key=self._calculate_fitness)
        return self._convert_to_assignment(best_solution)
    
    def _initialize_population(self) -> List[List[Tuple[str, int]]]:
        """Initialize population with random valid solutions."""
        population = []
        for _ in range(self.population_size):
            solution = []
            for delivery_idx in range(len(self.deliveries)):
                drone_idx = random.randint(0, len(self.drones) - 1)
                solution.append((self.drones[drone_idx].id, delivery_idx))
            population.append(solution)
        return population
    
    def _calculate_fitness(self, solution: List[Tuple[str, int]]) -> float:
        """Calculate fitness of a solution with load balancing and overuse penalty."""
        assignment = self._convert_to_assignment(solution)
        total_score = 0.0
        used_drones = set()
        delivery_counts = {drone.id: 0 for drone in self.drones}
        for drone_id, deliveries in assignment.items():
            drone = next(d for d in self.drones if d.id == drone_id)
            if deliveries:
                used_drones.add(drone_id)
            delivery_counts[drone_id] = len(deliveries)
            for delivery in deliveries:
                path = self.router.find_path(drone.current_position, delivery.position, drone, self.no_fly_zones, self.current_time)
                if not path or any(
                    zone.is_active(self.current_time) and any(
                        zone.intersects_line(path[i], path[i+1])
                        for i in range(len(path)-1)
                    ) for zone in self.no_fly_zones):
                    total_score -= 100.0  # Heavy penalty for invalid route
                    continue
                if delivery.status == "completed":
                    total_score += delivery.priority * 10.0
                distance = np.sqrt((drone.current_position[0] - delivery.position[0])**2 +
                                 (drone.current_position[1] - delivery.position[1])**2)
                energy_usage = distance / drone.speed
                total_score -= energy_usage * 2.0
                if not delivery.is_within_time_window(self.current_time):
                    total_score -= 20.0
        # Penalty for unused drones
        unused_drones = set(d.id for d in self.drones) - used_drones
        total_score -= len(unused_drones) * 50.0
        # Penalty for overused drones (quadratic penalty)
        for count in delivery_counts.values():
            if count > 0:
                total_score -= (count - len(self.deliveries) / len(self.drones)) ** 2 * 5
        # Bonus for using more drones
        total_score += len(used_drones) * 10.0
        return total_score
    
    def _select_parents(self, population: List[List[Tuple[str, int]]],
                       fitness_scores: List[float]) -> List[List[Tuple[str, int]]]:
        """Select parents using tournament selection."""
        parents = []
        for _ in range(len(population)):
            # Tournament selection
            candidates = random.sample(list(zip(population, fitness_scores)), 3)
            winner = max(candidates, key=lambda x: x[1])[0]
            parents.append(winner)
        return parents
    
    def _crossover(self, parent1: List[Tuple[str, int]],
                  parent2: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """Perform crossover between two parents."""
        point = random.randint(0, len(parent1) - 1)
        child = parent1[:point] + parent2[point:]
        return child
    
    def _mutate(self, solution: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """Apply mutation to a solution (higher mutation rate)."""
        mutation_rate = 0.3
        for idx in range(len(solution)):
            if random.random() < mutation_rate:
                new_drone_idx = random.randint(0, len(self.drones) - 1)
                solution[idx] = (self.drones[new_drone_idx].id, solution[idx][1])
        return solution
    
    def _random_individual(self) -> List[Tuple[str, int]]:
        """Create a random individual for diversity."""
        return [(random.choice(self.drones).id, i) for i in range(len(self.deliveries))]
    
    def _convert_to_assignment(self, solution: List[Tuple[str, int]]) -> Dict[str, List[Delivery]]:
        """Convert solution to delivery assignment."""
        assignment = {drone.id: [] for drone in self.drones}
        for drone_id, delivery_idx in solution:
            assignment[drone_id].append(self.deliveries[delivery_idx])
        return assignment

    def solve_greedy(self) -> Dict[str, List[Delivery]]:
        """Greedy fallback: assign each delivery to the nearest available drone."""
        assignment = {drone.id: [] for drone in self.drones}
        for delivery in self.deliveries:
            best_drone = None
            best_dist = float('inf')
            for drone in self.drones:
                if drone.can_carry(delivery.weight) and delivery.is_within_time_window(self.current_time):
                    path = self.router.find_path(drone.current_position, delivery.position, drone, self.no_fly_zones, self.current_time)
                    if not path or any(
                        zone.is_active(self.current_time) and any(
                            zone.intersects_line(path[i], path[i+1])
                            for i in range(len(path)-1)
                        ) for zone in self.no_fly_zones):
                        continue
                    dist = np.sqrt((drone.current_position[0] - delivery.position[0])**2 +
                                 (drone.current_position[1] - delivery.position[1])**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_drone = drone
            if best_drone:
                assignment[best_drone.id].append(delivery)
            else:
                delivery.mark_failed()
        return assignment 