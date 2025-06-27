from typing import List, Dict, Tuple, Set
import random
import numpy as np
from datetime import datetime, timedelta
from drone import Drone
from delivery import Delivery
from zone import NoFlyZone
from routing import AStarRouter
import time

class DataManager:
    """Drone, Delivery ve No-Fly Zone verilerini yönetmek için yardımcı sınıf"""
    
    @staticmethod
    def reset_drones(drones: List[Drone], initial_states: Dict = None):
        """Drone'ları başlangıç durumuna sıfırla"""
        for drone in drones:
            if initial_states and drone.id in initial_states:
                state = initial_states[drone.id]
                drone.current_position = list(state['position'])
                drone.current_battery = state['battery']
                drone.route = []
                if hasattr(drone, 'status'):
                    drone.status = state.get('status', 'available')
            else:
                # Varsayılan sıfırlama
                drone.route = []
                if hasattr(drone, 'status'):
                    drone.status = 'available'
    
    @staticmethod
    def reset_deliveries(deliveries: List[Delivery], initial_states: Dict = None):
        """Delivery'leri başlangıç durumuna sıfırla"""
        for delivery in deliveries:
            if initial_states and delivery.id in initial_states:
                state = initial_states[delivery.id]
                if hasattr(delivery, 'status'):
                    delivery.status = state.get('status', 'pending')
                if hasattr(delivery, 'assigned_drone'):
                    delivery.assigned_drone = state.get('assigned_drone', None)
            else:
                # Varsayılan sıfırlama
                if hasattr(delivery, 'status'):
                    delivery.status = 'pending'
                if hasattr(delivery, 'assigned_drone'):
                    delivery.assigned_drone = None
    
    @staticmethod
    def clear_all_lists(drones: List[Drone], deliveries: List[Delivery], 
                       no_fly_zones: List[NoFlyZone]):
        """Tüm listeleri temizle"""
        drones.clear()
        deliveries.clear()
        no_fly_zones.clear()
        print("[DATA_MANAGER] Tüm listeler temizlendi")
    
    @staticmethod
    def get_status_summary(drones: List[Drone], deliveries: List[Delivery], 
                          no_fly_zones: List[NoFlyZone]):
        """Mevcut durumun özetini döndür"""
        summary = {
            'total_drones': len(drones),
            'total_deliveries': len(deliveries),
            'total_no_fly_zones': len(no_fly_zones),
            'drone_statuses': {},
            'delivery_statuses': {}
        }
        
        # Drone durumları
        for drone in drones:
            status = getattr(drone, 'status', 'available')
            summary['drone_statuses'][status] = summary['drone_statuses'].get(status, 0) + 1
        
        # Delivery durumları
        for delivery in deliveries:
            status = getattr(delivery, 'status', 'pending')
            summary['delivery_statuses'][status] = summary['delivery_statuses'].get(status, 0) + 1
        
        return summary
    
    @staticmethod
    def print_status_summary(drones: List[Drone], deliveries: List[Delivery], 
                           no_fly_zones: List[NoFlyZone]):
        """Durumu konsola yazdır"""
        summary = DataManager.get_status_summary(drones, deliveries, no_fly_zones)
        
        print("\n" + "="*50)
        print("SISTEM DURUMU")
        print("="*50)
        print(f"Toplam Drone: {summary['total_drones']}")
        print(f"Toplam Delivery: {summary['total_deliveries']}")
        print(f"Toplam No-Fly Zone: {summary['total_no_fly_zones']}")
        
        if summary['drone_statuses']:
            print("\nDrone Durumları:")
            for status, count in summary['drone_statuses'].items():
                print(f"  {status}: {count}")
        
        if summary['delivery_statuses']:
            print("\nDelivery Durumları:")
            for status, count in summary['delivery_statuses'].items():
                print(f"  {status}: {count}")
        
        print("="*50 + "\n")

class DeliveryOptimizer:
    def __init__(self, drones: List[Drone], deliveries: List[Delivery], 
                 no_fly_zones: List[NoFlyZone], current_time: datetime):
        if not drones or not deliveries:
            raise ValueError("DeliveryOptimizer: Drone ve teslimat listeleri boş olamaz!")
        self.drones = drones
        self.deliveries = sorted(deliveries)  # Sort by priority and deadline
        self.no_fly_zones = no_fly_zones
        self.current_time = current_time
        self.assignment: Dict[str, List[Delivery]] = {drone.id: [] for drone in drones}
        self.router = AStarRouter((100, 100))  # Use default grid size or pass as needed
        
        # Başlangıç durumlarını sakla (reset için)
        self._store_initial_states()
    
    def _store_initial_states(self):
        """Başlangıç durumlarını sakla - reset için kullanılacak"""
        self.initial_drone_states = {}
        for drone in self.drones:
            self.initial_drone_states[drone.id] = {
                'position': tuple(drone.current_position),  # tuple olarak sakla
                'battery': drone.current_battery,
                'route': [],
                'status': getattr(drone, 'status', 'available')
            }
        
        self.initial_delivery_states = {}
        for delivery in self.deliveries:
            self.initial_delivery_states[delivery.id] = {
                'status': getattr(delivery, 'status', 'pending'),
                'assigned_drone': getattr(delivery, 'assigned_drone', None)
            }
    
    def reset_all_data(self):
        """Tüm drone'ları, delivery'leri ve assignment'ları başlangıç durumuna sıfırla"""
        print("[RESET] Tüm veriler sıfırlanıyor...")
        
        # Drone'ları sıfırla
        for drone in self.drones:
            if drone.id in self.initial_drone_states:
                initial_state = self.initial_drone_states[drone.id]
                drone.current_position = list(initial_state['position'])
                drone.current_battery = initial_state['battery']
                drone.route = []
                if hasattr(drone, 'status'):
                    drone.status = initial_state['status']
        
        # Delivery'leri sıfırla
        for delivery in self.deliveries:
            if delivery.id in self.initial_delivery_states:
                initial_state = self.initial_delivery_states[delivery.id]
                if hasattr(delivery, 'status'):
                    delivery.status = initial_state['status']
                if hasattr(delivery, 'assigned_drone'):
                    delivery.assigned_drone = initial_state['assigned_drone']
        
        # Assignment'ları sıfırla
        self.assignment = {drone.id: [] for drone in self.drones}
        
        print(f"[RESET] {len(self.drones)} drone, {len(self.deliveries)} delivery sıfırlandı")
        return True
    
    def clear_all_data(self):
        """Tüm verileri tamamen temizle (listeler boş olur)"""
        print("[CLEAR] Tüm veriler temizleniyor...")
        
        self.drones.clear()
        self.deliveries.clear()
        self.no_fly_zones.clear()
        self.assignment.clear()
        self.initial_drone_states.clear()
        self.initial_delivery_states.clear()
        
        print("[CLEAR] Tüm veriler temizlendi")
        return True
    
    def add_drone(self, drone: Drone):
        """Yeni drone ekle ve başlangıç durumunu kaydet"""
        self.drones.append(drone)
        self.assignment[drone.id] = []
        
        # Başlangıç durumunu kaydet
        self.initial_drone_states[drone.id] = {
            'position': tuple(drone.current_position),
            'battery': drone.current_battery,
            'route': [],
            'status': getattr(drone, 'status', 'available')
        }
        print(f"[ADD] Drone {drone.id} eklendi")
    
    def add_delivery(self, delivery: Delivery):
        """Yeni delivery ekle ve başlangıç durumunu kaydet"""
        self.deliveries.append(delivery)
        self.deliveries = sorted(self.deliveries)  # Priority'ye göre sırala
        
        # Başlangıç durumunu kaydet
        self.initial_delivery_states[delivery.id] = {
            'status': getattr(delivery, 'status', 'pending'),
            'assigned_drone': getattr(delivery, 'assigned_drone', None)
        }
        print(f"[ADD] Delivery {delivery.id} eklendi")
    
    def add_no_fly_zone(self, zone: NoFlyZone):
        """Yeni no-fly zone ekle"""
        self.no_fly_zones.append(zone)
        print(f"[ADD] No-fly zone eklendi")
    
    def remove_drone(self, drone_id: str):
        """Drone'u sil"""
        self.drones = [d for d in self.drones if d.id != drone_id]
        if drone_id in self.assignment:
            del self.assignment[drone_id]
        if drone_id in self.initial_drone_states:
            del self.initial_drone_states[drone_id]
        print(f"[REMOVE] Drone {drone_id} silindi")
    
    def remove_delivery(self, delivery_id: str):
        """Delivery'yi sil"""
        self.deliveries = [d for d in self.deliveries if d.id != delivery_id]
        if delivery_id in self.initial_delivery_states:
            del self.initial_delivery_states[delivery_id]
        print(f"[REMOVE] Delivery {delivery_id} silindi")
    
    def solve_csp(self, timeout_seconds: float = 30.0) -> Dict[str, List[Delivery]]:
        """Improved CSP: Assign deliveries to drones in sequence, maximizing completed deliveries. Robust against infinite loops and excessive slowness."""
        start_time = time.time()
        deliveries = sorted(self.deliveries, key=lambda d: (-d.priority, d.time_window_start))
        drone_states = {drone.id: {
            'position': drone.current_position,
            'battery': drone.current_battery,
            'time': self.current_time,
            'route': list(drone.route)
        } for drone in self.drones}
        assignment = {drone.id: [] for drone in self.drones}
        for delivery in deliveries:
            if time.time() - start_time > timeout_seconds:
                print(f"[CSP] Timeout reached ({timeout_seconds}s), returning partial solution")
                break
            best_drone = None
            best_arrival_time = None
            best_path = None
            # Path-finding ve uygunluk kontrollerini optimize et
            for drone in self.drones:
                state = drone_states[drone.id]
                if delivery.weight > drone.max_weight:
                    continue
                # Path-finding'i try/except ile güvenli yap
                try:
                    path = self.router.find_path(state['position'], delivery.position, drone, self.no_fly_zones, state['time'])
                except Exception as e:
                    print(f"[CSP] Path finding error: {e}")
                    continue
                if not path or len(path) < 2:
                    continue
                # No-fly zone ve path intersect kontrolünü optimize et
                path_blocked = False
                for zone in self.no_fly_zones:
                    if zone.is_active(state['time']):
                        for i in range(len(path)-1):
                            if zone.intersects_line(path[i], path[i+1]):
                                path_blocked = True
                                break
                        if path_blocked:
                            break
                if path_blocked:
                    continue
                total_dist = sum(np.sqrt((path[i+1][0] - path[i][0])**2 + (path[i+1][1] - path[i][1])**2) for i in range(len(path)-1))
                travel_time = total_dist / drone.speed
                arrival_time = state['time'] + timedelta(hours=travel_time)
                battery_used = total_dist
                if battery_used > state['battery']:
                    continue
                if not (delivery.time_window_start <= arrival_time <= delivery.time_window_end):
                    continue
                if best_arrival_time is None or arrival_time < best_arrival_time:
                    best_drone = drone
                    best_arrival_time = arrival_time
                    best_path = path
            if best_drone:
                assignment[best_drone.id].append(delivery)
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
                 population_size: int = 30, generations: int = 20, max_time: float = 20.0, early_stop_rounds: int = 5):
        if not drones or not deliveries:
            raise ValueError("GeneticOptimizer: Drone ve teslimat listeleri boş olamaz!")
        self.drones = drones
        self.deliveries = deliveries
        self.no_fly_zones = no_fly_zones
        self.current_time = current_time
        self.population_size = population_size
        self.generations = generations
        self.router = AStarRouter((100, 100))
        self.max_time = max_time
        self.early_stop_rounds = early_stop_rounds
        
        # Başlangıç durumlarını sakla (reset için)
        self._store_initial_states()
    
    def _store_initial_states(self):
        """Başlangıç durumlarını sakla - reset için kullanılacak"""
        self.initial_drone_states = {}
        for drone in self.drones:
            self.initial_drone_states[drone.id] = {
                'position': tuple(drone.current_position),
                'battery': drone.current_battery,
                'route': [],
                'status': getattr(drone, 'status', 'available')
            }
        
        self.initial_delivery_states = {}
        for delivery in self.deliveries:
            self.initial_delivery_states[delivery.id] = {
                'status': getattr(delivery, 'status', 'pending'),
                'assigned_drone': getattr(delivery, 'assigned_drone', None)
            }
    
    def reset_all_data(self):
        """Tüm drone'ları, delivery'leri başlangıç durumuna sıfırla"""
        print("[GA-RESET] Tüm veriler sıfırlanıyor...")
        
        # Drone'ları sıfırla
        for drone in self.drones:
            if drone.id in self.initial_drone_states:
                initial_state = self.initial_drone_states[drone.id]
                drone.current_position = list(initial_state['position'])
                drone.current_battery = initial_state['battery']
                drone.route = []
                if hasattr(drone, 'status'):
                    drone.status = initial_state['status']
        
        # Delivery'leri sıfırla
        for delivery in self.deliveries:
            if delivery.id in self.initial_delivery_states:
                initial_state = self.initial_delivery_states[delivery.id]
                if hasattr(delivery, 'status'):
                    delivery.status = initial_state['status']
                if hasattr(delivery, 'assigned_drone'):
                    delivery.assigned_drone = initial_state['assigned_drone']
        
        print(f"[GA-RESET] {len(self.drones)} drone, {len(self.deliveries)} delivery sıfırlandı")
        return True
    
    def solve(self, timeout_seconds: float = 30.0) -> Dict[str, List[Delivery]]:
        """Solve delivery assignment using Genetic Algorithm with early stopping, time limit, and 30s timeout."""
        start_time = time.time()
        population = self._initialize_population()
        best_fitness = float('-inf')
        best_solution = None
        no_improve_rounds = 0
        timeout_printed = False
        
        # Her 5 generasyonda bir timeout kontrolü yap
        timeout_check_interval = 5
        
        for generation in range(self.generations):
            # Her iterasyonda timeout kontrolü
            current_time = time.time()
            if current_time - start_time > timeout_seconds:
                if not timeout_printed:
                    print(f"[GA] Timeout reached ({timeout_seconds:.1f}s) at generation {generation}, returning best solution found")
                    timeout_printed = True
                break
            # Fitness hesaplamasında da timeout kontrolü
            fitness_start = time.time()
            fitness_scores = []
            for i, solution in enumerate(population):
                if time.time() - start_time > timeout_seconds:
                    print(f"[GA] Timeout during fitness calculation at generation {generation}, using partial results")
                    # Kalan çözümler için düşük fitness değeri ata
                    fitness_scores.extend([float('-inf')] * (len(population) - i))
                    break
                fitness_scores.append(self._calculate_fitness(solution))
            
            if not fitness_scores:  # Eğer hiç fitness hesaplanamadıysa
                break
                
            gen_best = max(fitness_scores) if fitness_scores else float('-inf')
            if gen_best > best_fitness:
                best_fitness = gen_best
                best_solution = population[fitness_scores.index(gen_best)] if fitness_scores else None
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1
            
            # Early stopping kontrolü
            if no_improve_rounds >= self.early_stop_rounds:
                print(f"[GA] Early stopping at generation {generation} (no improvement for {self.early_stop_rounds} rounds)")
                break
            
            # Timeout kontrolü - parent selection öncesi
            if time.time() - start_time > timeout_seconds:
                print(f"[GA] Timeout before parent selection at generation {generation}")
                break
            parents = self._select_parents(population, fitness_scores)
            new_population = []
            
            # Yeni popülasyon oluştururken timeout kontrolü
            while len(new_population) < self.population_size:
                if time.time() - start_time > timeout_seconds:
                    print(f"[GA] Timeout during population generation at generation {generation}")
                    # Mevcut popülasyonla devam et
                    break
                    
                if random.random() < 0.2:  # 20% chance to inject random individual
                    new_population.append(self._random_individual())
                else:
                    parent1, parent2 = random.sample(parents, 2)
                    child = self._crossover(parent1, parent2)
                    child = self._mutate(child)
                    new_population.append(child)
            
            # Eğer yeni popülasyon tam oluşturulamadıysa, eskisini kullan
            if len(new_population) < self.population_size:
                population = new_population + population[:(self.population_size - len(new_population))]
            else:
                population = new_population
        
        # En iyi çözümü döndür
        if best_solution is None:
            if population:
                try:
                    best_solution = max(population, key=self._calculate_fitness)
                except:
                    # Fitness hesaplamasında hata varsa ilk çözümü kullan
                    best_solution = population[0] if population else self._random_individual()
            else:
                best_solution = self._random_individual()
        
        print(f"[GA] Algorithm completed in {time.time() - start_time:.2f} seconds")
        assignment = self._convert_to_assignment(best_solution)
        
        # Son adımda da timeout kontrolü yaparak güvenli bir şekilde sonlandır
        assignment_start = time.time()
        for drone_id, deliveries in assignment.items():
            if time.time() - start_time > timeout_seconds:
                print(f"[GA] Timeout during final assignment processing")
                break
                
            drone = next(d for d in self.drones if d.id == drone_id)
            for delivery in deliveries:
                if time.time() - start_time > timeout_seconds:
                    print(f"[GA] Timeout during delivery processing")
                    break
                    
                try:
                    path = self.router.find_path(drone.current_position, delivery.position, drone, self.no_fly_zones, self.current_time)
                except Exception as e:
                    print(f"[GA] Path finding error for delivery {delivery.id}: {e}")
                    path = []
                if not path:
                    print(f"[GA] No path found for delivery {delivery.id} (drone {drone.id})")
                elif any(
                    zone.is_active(self.current_time) and any(
                        zone.intersects_line(path[i], path[i+1])
                        for i in range(len(path)-1)
                    ) for zone in self.no_fly_zones):
                    print(f"[GA] Path blocked by no-fly zone for delivery {delivery.id} (drone {drone.id})")
                    delivery.mark_failed()
                else:
                    delivery.mark_completed()
        return assignment

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
        """Calculate fitness of a solution with load balancing and overuse penalty. Robust against path-finding errors."""
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
                try:
                    path = self.router.find_path(drone.current_position, delivery.position, drone, self.no_fly_zones, self.current_time)
                except Exception as e:
                    path = []
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
        """Greedy fallback: assign each delivery to the nearest available drone. Robust against path-finding errors."""
        assignment = {drone.id: [] for drone in self.drones}
        for delivery in self.deliveries:
            best_drone = None
            best_dist = float('inf')
            for drone in self.drones:
                if drone.can_carry(delivery.weight) and delivery.is_within_time_window(self.current_time):
                    try:
                        path = self.router.find_path(drone.current_position, delivery.position, drone, self.no_fly_zones, self.current_time)
                    except Exception as e:
                        path = []
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
                delivery.mark_completed()
            else:
                delivery.mark_failed()
        return assignment