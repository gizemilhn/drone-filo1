import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QTabWidget, QPushButton, QLabel, 
                            QLineEdit, QSpinBox, QDoubleSpinBox, QTableWidget, 
                            QTableWidgetItem, QMessageBox, QGroupBox, QFormLayout,
                            QCheckBox, QDialog, QDialogButtonBox, QRadioButton,
                            QButtonGroup)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QPolygonF
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from datetime import datetime, timedelta
import concurrent.futures

from drone import Drone
from delivery import Delivery
from zone import NoFlyZone
from routing import AStarRouter
from optimizer import DeliveryOptimizer, GeneticOptimizer
from main import DroneDeliverySystem

class MapCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MapCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        # Initialize empty data
        self.drones = []
        self.deliveries = []
        self.no_fly_zones = []
        self.routes = {}
        
    def update_plot(self, drones, deliveries, no_fly_zones, routes=None):
        self.axes.clear()
        self.drones = drones
        self.deliveries = deliveries
        self.no_fly_zones = no_fly_zones
        if routes:
            self.routes = routes
            
        # Plot no-fly zones
        for zone in no_fly_zones:
            polygon = plt.Polygon(zone.polygon_coordinates, 
                                facecolor='red', alpha=0.3,
                                label='No-Fly Zone')
            self.axes.add_patch(polygon)
            
        # Plot delivery points
        for delivery in deliveries:
            color = 'green' if delivery.status == "completed" else \
                   'red' if delivery.status == "failed" else \
                   'blue' if delivery.status == "in_progress" else 'gray'
            
            self.axes.scatter(delivery.position[0], delivery.position[1],
                            c=color, marker='o', s=100,
                            label=f'Delivery {delivery.id}')
            
            # Add priority label
            self.axes.text(delivery.position[0], delivery.position[1] + 0.5,
                          f'P{delivery.priority}', ha='center')
            
        # Plot drones and their routes
        colors = plt.cm.rainbow(np.linspace(0, 1, len(drones)))
        for i, drone in enumerate(drones):
            color = colors[i]
            
            # Plot drone's current position
            self.axes.scatter(drone.current_position[0], drone.current_position[1],
                            c=[color], marker='^', s=150,
                            label=f'Drone {drone.id}')
            
            # Plot drone's route
            if drone.id in self.routes:
                route = self.routes[drone.id]
                route_x = [p[0] for p in route]
                route_y = [p[1] for p in route]
                self.axes.plot(route_x, route_y, c=color, alpha=0.7, linestyle='-', linewidth=2)
            
            # Add battery level
            battery_text = f'Battery: {drone.get_remaining_battery_percentage():.1f}%'
            self.axes.text(drone.current_position[0], drone.current_position[1] - 0.5,
                          battery_text, ha='center', fontsize=8)
            
        self.axes.set_xlim(0, 100)
        self.axes.set_ylim(0, 100)
        self.axes.set_title("Drone Delivery Fleet Visualization")
        self.axes.set_xlabel("X Coordinate")
        self.axes.set_ylabel("Y Coordinate")
        self.axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        self.fig.tight_layout()
        self.draw()

class ReportPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        
        # Statistics group
        stats_group = QGroupBox("Delivery Statistics")
        stats_layout = QFormLayout()
        self.total_deliveries = QLabel("0")
        self.completed_deliveries = QLabel("0")
        self.failed_deliveries = QLabel("0")
        self.in_progress_deliveries = QLabel("0")
        
        stats_layout.addRow("Total Deliveries:", self.total_deliveries)
        stats_layout.addRow("Completed:", self.completed_deliveries)
        stats_layout.addRow("Failed:", self.failed_deliveries)
        stats_layout.addRow("In Progress:", self.in_progress_deliveries)
        stats_group.setLayout(stats_layout)
        
        # Drone statistics table
        self.drone_table = QTableWidget()
        self.drone_table.setColumnCount(4)
        self.drone_table.setHorizontalHeaderLabels(["Drone ID", "Battery", "Distance", "Deliveries"])
        
        layout.addWidget(stats_group)
        layout.addWidget(QLabel("Drone Statistics:"))
        layout.addWidget(self.drone_table)
        self.setLayout(layout)
        
    def update_report(self, report):
        self.total_deliveries.setText(str(report['total_deliveries']))
        self.completed_deliveries.setText(str(report['completed_deliveries']))
        self.failed_deliveries.setText(str(report['failed_deliveries']))
        self.in_progress_deliveries.setText(str(report['in_progress_deliveries']))
        
        # Update drone table
        self.drone_table.setRowCount(len(report['drone_statistics']))
        for i, (drone_id, stats) in enumerate(report['drone_statistics'].items()):
            self.drone_table.setItem(i, 0, QTableWidgetItem(drone_id))
            self.drone_table.setItem(i, 1, QTableWidgetItem(f"{stats['battery_remaining']:.1f}%"))
            self.drone_table.setItem(i, 2, QTableWidgetItem(f"{stats['distance_traveled']:.1f}"))
            self.drone_table.setItem(i, 3, QTableWidgetItem(str(stats['deliveries_completed'])))

class OptimizerThread(QThread):
    finished = pyqtSignal(dict)
    def __init__(self, system, use_genetic, use_greedy=False):
        super().__init__()
        self.system = system
        self.use_genetic = use_genetic
        self.use_greedy = use_greedy
    def run(self):
        try:
            assignment = self.system.optimize_deliveries(use_genetic=self.use_genetic, use_greedy=self.use_greedy)
        except Exception as e:
            print(f"[OptimizerThread] Exception: {e}")
            assignment = {}
        self.finished.emit(assignment)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone Delivery Fleet Optimization")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize system
        self.system = DroneDeliverySystem()
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Create left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Add buttons for adding new items
        add_drone_btn = QPushButton("Add New Drone")
        add_delivery_btn = QPushButton("Add New Delivery")
        add_zone_btn = QPushButton("Add New No-Fly Zone")
        
        add_drone_btn.clicked.connect(self.show_add_drone_dialog)
        add_delivery_btn.clicked.connect(self.show_add_delivery_dialog)
        add_zone_btn.clicked.connect(self.show_add_zone_dialog)
        
        # Add load sample data button
        load_sample_btn = QPushButton("Load Sample Data")
        load_sample_btn.clicked.connect(self.load_sample_data)
        
        left_layout.addWidget(add_drone_btn)
        left_layout.addWidget(add_delivery_btn)
        left_layout.addWidget(add_zone_btn)
        left_layout.addWidget(load_sample_btn)
        
        # Add optimization controls
        optimize_group = QGroupBox("Optimization")
        optimize_layout = QVBoxLayout()
        
        optimize_btn = QPushButton("Optimize Deliveries")
        optimize_btn.clicked.connect(self.optimize_deliveries)
        
        optimize_layout.addWidget(optimize_btn)
        optimize_group.setLayout(optimize_layout)
        
        left_layout.addWidget(optimize_group)
        
        # Algorithm selection
        self.csp_radio = QRadioButton("CSP (Recommended)")
        self.ga_radio = QRadioButton("Genetic Algorithm")
        self.greedy_radio = QRadioButton("Greedy (Fastest)")
        self.csp_radio.setChecked(True)
        self.alg_group = QButtonGroup()
        self.alg_group.addButton(self.csp_radio)
        self.alg_group.addButton(self.ga_radio)
        self.alg_group.addButton(self.greedy_radio)
        left_layout.addWidget(self.csp_radio)
        left_layout.addWidget(self.ga_radio)
        left_layout.addWidget(self.greedy_radio)
        
        # Add map canvas
        self.map_canvas = MapCanvas()
        
        # Add report panel
        self.report_panel = ReportPanel()
        
        # Add widgets to main layout
        layout.addWidget(left_panel, 1)
        layout.addWidget(self.map_canvas, 3)
        layout.addWidget(self.report_panel, 1)
        
        # Update visualization
        self.update_visualization()
        
        self.optimizing_dialog = None
        self.optimizer_thread = None
        
    def show_add_drone_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Add New Drone")
        layout = QFormLayout()
        
        id_input = QLineEdit()
        max_weight = QDoubleSpinBox()
        max_weight.setRange(0, 100)
        battery = QDoubleSpinBox()
        battery.setRange(0, 100000)
        speed = QDoubleSpinBox()
        speed.setRange(0, 100)
        start_x = QDoubleSpinBox()
        start_x.setRange(0, 100)
        start_y = QDoubleSpinBox()
        start_y.setRange(0, 100)
        
        layout.addRow("ID:", id_input)
        layout.addRow("Max Weight:", max_weight)
        layout.addRow("Battery Capacity:", battery)
        layout.addRow("Speed:", speed)
        layout.addRow("Start X:", start_x)
        layout.addRow("Start Y:", start_y)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted:
            drone = Drone(
                id_input.text(),
                max_weight.value(),
                battery.value(),
                speed.value(),
                (start_x.value(), start_y.value())
            )
            self.system.add_drone(drone)
            self.update_visualization()
            
    def show_add_delivery_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Add New Delivery")
        layout = QFormLayout()
        
        id_input = QLineEdit()
        pos_x = QDoubleSpinBox()
        pos_x.setRange(0, 100)
        pos_y = QDoubleSpinBox()
        pos_y.setRange(0, 100)
        weight = QDoubleSpinBox()
        weight.setRange(0, 100)
        priority = QSpinBox()
        priority.setRange(1, 5)
        
        layout.addRow("ID:", id_input)
        layout.addRow("Position X:", pos_x)
        layout.addRow("Position Y:", pos_y)
        layout.addRow("Weight:", weight)
        layout.addRow("Priority:", priority)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted:
            current_time = datetime.now()
            delivery = Delivery(
                id_input.text(),
                (pos_x.value(), pos_y.value()),
                weight.value(),
                priority.value(),
                current_time,
                current_time + timedelta(hours=1)
            )
            self.system.add_delivery(delivery)
            self.update_visualization()
            
    def show_add_zone_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Add New No-Fly Zone")
        layout = QFormLayout()
        
        id_input = QLineEdit()
        coords_input = QLineEdit()
        coords_input.setPlaceholderText("Format: x1,y1;x2,y2;x3,y3;x4,y4")
        
        layout.addRow("ID:", id_input)
        layout.addRow("Coordinates:", coords_input)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted:
            try:
                coords = []
                for point in coords_input.text().split(';'):
                    x, y = map(float, point.split(','))
                    coords.append((x, y))
                
                current_time = datetime.now()
                zone = NoFlyZone(
                    id_input.text(),
                    coords,
                    current_time,
                    current_time + timedelta(hours=2)
                )
                self.system.add_no_fly_zone(zone)
                self.update_visualization()
            except:
                QMessageBox.warning(self, "Error", "Invalid coordinates format")
                
    def optimize_deliveries(self):
        # Show loading dialog
        self.optimizing_dialog = QMessageBox(self)
        self.optimizing_dialog.setWindowTitle("Optimizing...")
        self.optimizing_dialog.setText("Optimization in progress. Please wait...")
        self.optimizing_dialog.setStandardButtons(QMessageBox.NoButton)
        self.optimizing_dialog.show()
        # Determine algorithm
        use_genetic = self.ga_radio.isChecked()
        use_greedy = self.greedy_radio.isChecked()
        self.optimizer_thread = OptimizerThread(self.system, use_genetic=use_genetic, use_greedy=use_greedy)
        self.optimizer_thread.finished.connect(self.on_optimization_finished)
        self.optimizer_thread.start()

    def on_optimization_finished(self, assignment):
        self.system.execute_deliveries(assignment)
        self.update_visualization()
        if self.optimizing_dialog:
            self.optimizing_dialog.close()
            self.optimizing_dialog = None
        self.optimizer_thread = None
        
    def update_visualization(self):
        # Update map
        routes = {}
        for drone in self.system.drones:
            if len(drone.route) > 1:
                routes[drone.id] = drone.route
        self.map_canvas.update_plot(
            self.system.drones,
            self.system.deliveries,
            self.system.no_fly_zones,
            routes
        )
        
        # Update report
        report = self.system.generate_report()
        self.report_panel.update_report(report)

    def load_sample_data(self):
        # Sample drones
        drones = [
            {"id": "1", "max_weight": 4.0, "battery": 12000, "speed": 8.0, "start_pos": (10, 10)},
            {"id": "2", "max_weight": 3.5, "battery": 10000, "speed": 10.0, "start_pos": (20, 30)},
            {"id": "3", "max_weight": 5.0, "battery": 15000, "speed": 7.0, "start_pos": (50, 50)},
            {"id": "4", "max_weight": 2.0, "battery": 8000, "speed": 12.0, "start_pos": (80, 20)},
            {"id": "5", "max_weight": 6.0, "battery": 20000, "speed": 5.0, "start_pos": (40, 70)}
        ]
        
        # Sample deliveries
        deliveries = [
            {"id": "1", "pos": (15, 25), "weight": 1.5, "priority": 3, "time_window": (0, 60)},
            {"id": "2", "pos": (30, 40), "weight": 2.0, "priority": 5, "time_window": (0, 30)},
            {"id": "3", "pos": (70, 80), "weight": 3.0, "priority": 2, "time_window": (20, 80)},
            {"id": "4", "pos": (90, 10), "weight": 1.0, "priority": 4, "time_window": (10, 40)},
            {"id": "5", "pos": (45, 60), "weight": 4.0, "priority": 1, "time_window": (30, 90)},
            {"id": "6", "pos": (25, 15), "weight": 2.5, "priority": 3, "time_window": (0, 50)},
            {"id": "7", "pos": (60, 30), "weight": 1.0, "priority": 5, "time_window": (5, 25)},
            {"id": "8", "pos": (85, 90), "weight": 3.5, "priority": 2, "time_window": (40, 100)},
            {"id": "9", "pos": (10, 80), "weight": 2.0, "priority": 4, "time_window": (15, 45)},
            {"id": "10", "pos": (95, 50), "weight": 1.5, "priority": 3, "time_window": (0, 60)},
            {"id": "11", "pos": (55, 20), "weight": 0.5, "priority": 5, "time_window": (0, 20)},
            {"id": "12", "pos": (35, 75), "weight": 2.0, "priority": 1, "time_window": (50, 120)},
            {"id": "13", "pos": (75, 40), "weight": 3.0, "priority": 3, "time_window": (10, 50)},
            {"id": "14", "pos": (20, 90), "weight": 1.5, "priority": 4, "time_window": (30, 70)},
            {"id": "15", "pos": (65, 65), "weight": 4.5, "priority": 2, "time_window": (25, 75)},
            {"id": "16", "pos": (40, 10), "weight": 2.0, "priority": 5, "time_window": (0, 30)},
            {"id": "17", "pos": (5, 50), "weight": 1.0, "priority": 3, "time_window": (15, 55)},
            {"id": "18", "pos": (50, 85), "weight": 3.0, "priority": 1, "time_window": (60, 100)},
            {"id": "19", "pos": (80, 70), "weight": 2.5, "priority": 4, "time_window": (20, 60)},
            {"id": "20", "pos": (30, 55), "weight": 1.5, "priority": 2, "time_window": (40, 80)}
        ]
        
        # Sample no-fly zones
        no_fly_zones = [
            {
                "id": "1",
                "coordinates": [(40, 30), (60, 30), (60, 50), (40, 50)],
                "active_time": (0, 120)
            },
            {
                "id": "2",
                "coordinates": [(70, 10), (90, 10), (90, 30), (70, 30)],
                "active_time": (30, 90)
            },
            {
                "id": "3",
                "coordinates": [(10, 60), (30, 60), (30, 80), (10, 80)],
                "active_time": (0, 60)
            }
        ]
        
        # Clear existing data
        self.system = DroneDeliverySystem()
        
        # Add drones
        for drone_data in drones:
            drone = Drone(
                str(drone_data["id"]),
                drone_data["max_weight"],
                drone_data["battery"],
                drone_data["speed"],
                drone_data["start_pos"]
            )
            self.system.add_drone(drone)
            
        # Add deliveries
        current_time = datetime.now()
        for delivery_data in deliveries:
            delivery = Delivery(
                str(delivery_data["id"]),
                delivery_data["pos"],
                delivery_data["weight"],
                delivery_data["priority"],
                current_time + timedelta(minutes=delivery_data["time_window"][0]),
                current_time + timedelta(minutes=delivery_data["time_window"][1])
            )
            self.system.add_delivery(delivery)
            
        # Add no-fly zones
        for zone_data in no_fly_zones:
            zone = NoFlyZone(
                str(zone_data["id"]),
                zone_data["coordinates"],
                current_time + timedelta(minutes=zone_data["active_time"][0]),
                current_time + timedelta(minutes=zone_data["active_time"][1])
            )
            self.system.add_no_fly_zone(zone)
            
        # Update visualization
        self.update_visualization()
        QMessageBox.information(self, "Success", "Sample data loaded successfully!")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 