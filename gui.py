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
from optimizer import DeliveryOptimizer, GeneticOptimizer, DataManager
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
        
        # Add individual remove buttons - standart stil
        remove_drone_btn = QPushButton("Remove Drone")
        remove_drone_btn.clicked.connect(self.show_remove_drone_dialog)
        
        remove_delivery_btn = QPushButton("Remove Delivery")
        remove_delivery_btn.clicked.connect(self.show_remove_delivery_dialog)
        
        remove_zone_btn = QPushButton("Remove No-Fly Zone")
        remove_zone_btn.clicked.connect(self.show_remove_zone_dialog)
        
        # Add reset and clear buttons - standart stil
        reset_data_btn = QPushButton("Reset All Data")
        reset_data_btn.clicked.connect(self.reset_all_data)
        
        clear_data_btn = QPushButton("Clear All Data")
        clear_data_btn.clicked.connect(self.clear_all_data)
        
        # Add show status button - standart stil
        show_status_btn = QPushButton("Show Status")
        show_status_btn.clicked.connect(self.show_system_status)
        
        left_layout.addWidget(add_drone_btn)
        left_layout.addWidget(add_delivery_btn)
        left_layout.addWidget(add_zone_btn)
        left_layout.addWidget(load_sample_btn)
        
        # Add individual remove buttons
        left_layout.addWidget(remove_drone_btn)
        left_layout.addWidget(remove_delivery_btn)
        left_layout.addWidget(remove_zone_btn)
        
        # Add separator
        separator = QLabel("─" * 20)
        separator.setAlignment(Qt.AlignCenter)
        separator.setStyleSheet("color: #888888; font-size: 10px;")
        left_layout.addWidget(separator)
        
        # Add reset and clear buttons
        left_layout.addWidget(reset_data_btn)
        left_layout.addWidget(clear_data_btn)
        left_layout.addWidget(show_status_btn)
        
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
        # Başarılı teslimat sayısı çok azsa veya hiç yoksa kullanıcıya uyarı göster
        completed = sum(1 for d in self.system.deliveries if d.status == "completed")
        total = len(self.system.deliveries)
        if completed < total // 3:  # Teslimatların üçte birinden azı başarılıysa uyarı ver
            QMessageBox.warning(self, "Optimizasyon Uyarısı", "Optimizasyonun büyük kısmı başarısız oldu.\nCSP algoritması karmaşık veriyle yavaş kalabilir.\nDaha hızlı sonuç için Greedy veya Genetic Algorithm seçebilirsiniz.")
        # Optimize dialogu açıkken, optimizasyon tamamlandığında mutlaka kapat
        if self.optimizing_dialog:
            self.optimizing_dialog.done(0)
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
    
    def reset_all_data(self):
        """Tüm verileri başlangıç durumuna sıfırla"""
        reply = QMessageBox.question(
            self, 
            "Reset All Data", 
            "Bu işlem tüm drone'ları, delivery'leri ve atamalarını başlangıç durumuna sıfırlayacak.\n"
            "Optimizasyon sonuçları kaybolacak. Devam etmek istiyor musunuz?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # System içindeki verileri sıfırla
                if hasattr(self.system, 'optimizer') and self.system.optimizer:
                    self.system.optimizer.reset_all_data()
                    QMessageBox.information(self, "Success", "Tüm veriler başlangıç durumuna sıfırlandı!")
                else:
                    # Manuel sıfırlama
                    from optimizer import DataManager
                    DataManager.reset_drones(self.system.drones)
                    DataManager.reset_deliveries(self.system.deliveries)
                    QMessageBox.information(self, "Success", "Tüm veriler başlangıç durumuna sıfırlandı!")
                
                # Görselleştirmeyi güncelle
                self.update_visualization()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Sıfırlama sırasında hata oluştu:\n{str(e)}")
    
    def clear_all_data(self):
        """Tüm verileri tamamen temizle"""
        reply = QMessageBox.question(
            self, 
            "Clear All Data", 
            "⚠️ DİKKAT ⚠️\n\n"
            "Bu işlem TÜM verileri kalıcı olarak silecek:\n"
            "• Tüm drone'lar\n"
            "• Tüm delivery'ler\n"
            "• Tüm no-fly zone'lar\n"
            "• Tüm optimizasyon sonuçları\n\n"
            "Bu işlem GERİ ALINAMAZ!\n"
            "Devam etmek istiyor musunuz?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # İkinci onay
            confirm = QMessageBox.question(
                self,
                "Final Confirmation",
                "Son onay: Tüm verileri silmek istediğinizden emin misiniz?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if confirm == QMessageBox.Yes:
                try:
                    # Tüm verileri temizle
                    self.system.drones.clear()
                    self.system.deliveries.clear()
                    self.system.no_fly_zones.clear()
                    
                    # Optimizer varsa onu da temizle
                    if hasattr(self.system, 'optimizer'):
                        self.system.optimizer = None
                    
                    # Görselleştirmeyi güncelle
                    self.update_visualization()
                    
                    QMessageBox.information(self, "Success", "Tüm veriler temizlendi!")
                    
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Temizleme sırasında hata oluştu:\n{str(e)}")
    
    def show_system_status(self):
        """Sistem durumunu göster"""
        try:
            from optimizer import DataManager
            
            # Durumu al
            summary = DataManager.get_status_summary(
                self.system.drones, 
                self.system.deliveries, 
                self.system.no_fly_zones
            )
            
            # Detaylı bilgi hazırla
            status_text = "🤖 DRONE DELIVERY SYSTEM STATUS 🚁\n\n"
            status_text += "=" * 50 + "\n"
            status_text += f"📊 GENEL BİLGİLER\n"
            status_text += f"Toplam Drone: {summary['total_drones']}\n"
            status_text += f"Toplam Delivery: {summary['total_deliveries']}\n"
            status_text += f"Toplam No-Fly Zone: {summary['total_no_fly_zones']}\n\n"
            
            if summary['drone_statuses']:
                status_text += f"🚁 DRONE DURUMLARI\n"
                for status, count in summary['drone_statuses'].items():
                    emoji = "✅" if status == "available" else "🔄" if status == "busy" else "⚠️"
                    status_text += f"{emoji} {status.title()}: {count}\n"
                status_text += "\n"
            
            if summary['delivery_statuses']:
                status_text += f"📦 DELIVERY DURUMLARI\n"
                for status, count in summary['delivery_statuses'].items():
                    emoji = "⏳" if status == "pending" else "✅" if status == "completed" else "❌" if status == "failed" else "🔄"
                    status_text += f"{emoji} {status.title()}: {count}\n"
                status_text += "\n"
            
            # Detaylı drone bilgileri
            if self.system.drones:
                status_text += f"🔧 DETAYLI DRONE BİLGİLERİ\n"
                for drone in self.system.drones:
                    battery_emoji = "🔋" if drone.current_battery > 50 else "🪫" if drone.current_battery > 20 else "⚡"
                    status_text += f"{battery_emoji} {drone.id}: "
                    status_text += f"Pos({drone.current_position[0]:.1f},{drone.current_position[1]:.1f}) "
                    status_text += f"Battery:{drone.current_battery:.1f}% "
                    status_text += f"Load:{drone.max_weight}kg\n"
                status_text += "\n"
            
            # Optimizasyon durumu
            if hasattr(self.system, 'optimizer') and self.system.optimizer:
                status_text += f"⚙️ OPTİMİZASYON DURUMU\n"
                status_text += f"✅ Optimizer aktif\n"
                if hasattr(self.system.optimizer, 'assignment'):
                    total_assignments = sum(len(deliveries) for deliveries in self.system.optimizer.assignment.values())
                    status_text += f"📋 Toplam atama: {total_assignments}\n"
            else:
                status_text += f"⚙️ OPTİMİZASYON DURUMU\n"
                status_text += f"❌ Optimizer henüz çalıştırılmadı\n"
            
            status_text += "\n" + "=" * 50
            
            # Mesaj kutusunda göster
            msg = QMessageBox(self)
            msg.setWindowTitle("System Status")
            msg.setText(status_text)
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            
            # Mesaj kutusunu büyüt
            msg.setStyleSheet("""
                QMessageBox {
                    font-family: 'Courier New', monospace;
                    font-size: 10px;
                }
                QMessageBox QLabel {
                    min-width: 500px;
                    max-width: 500px;
                }
            """)
            
            msg.exec_()
            
            # Konsola da yazdır
            DataManager.print_status_summary(
                self.system.drones, 
                self.system.deliveries, 
                self.system.no_fly_zones
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Durum gösterilirken hata oluştu:\n{str(e)}")
    
    def show_remove_drone_dialog(self):
        """Drone seçip silme dialogu"""
        if not self.system.drones:
            QMessageBox.information(self, "Info", "Silinecek drone yok!")
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Remove Drone")
        dialog.setModal(True)
        dialog.resize(400, 300)
        
        layout = QVBoxLayout()
        
        # Açıklama
        info_label = QLabel("Silmek istediğiniz drone'u seçin:")
        info_label.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        # Drone listesi
        drone_list = QTableWidget()
        drone_list.setColumnCount(4)
        drone_list.setHorizontalHeaderLabels(["ID", "Position", "Battery", "Max Weight"])
        drone_list.setRowCount(len(self.system.drones))
        drone_list.setSelectionBehavior(QTableWidget.SelectRows)
        
        for i, drone in enumerate(self.system.drones):
            drone_list.setItem(i, 0, QTableWidgetItem(drone.id))
            drone_list.setItem(i, 1, QTableWidgetItem(f"({drone.current_position[0]:.1f}, {drone.current_position[1]:.1f})"))
            drone_list.setItem(i, 2, QTableWidgetItem(f"{drone.current_battery:.1f}%"))
            drone_list.setItem(i, 3, QTableWidgetItem(f"{drone.max_weight:.1f}kg"))
        
        layout.addWidget(drone_list)
        
        # Butonlar - standart stil
        button_layout = QHBoxLayout()
        
        remove_btn = QPushButton("Remove Selected")
        cancel_btn = QPushButton("Cancel")
        
        def remove_selected():
            selected_rows = drone_list.selectionModel().selectedRows()
            if not selected_rows:
                QMessageBox.warning(dialog, "Warning", "Lütfen silinecek drone'u seçin!")
                return
            
            row = selected_rows[0].row()
            drone = self.system.drones[row]
            
            reply = QMessageBox.question(
                dialog,
                "Confirm Remove",
                f"'{drone.id}' drone'unu silmek istediğinizden emin misiniz?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                try:
                    # System'den sil
                    self.system.drones.remove(drone)
                    
                    # Optimizer varsa oradan da sil
                    if hasattr(self.system, 'optimizer') and self.system.optimizer:
                        self.system.optimizer.remove_drone(drone.id)
                    
                    # Görselleştirmeyi güncelle
                    self.update_visualization()
                    
                    QMessageBox.information(dialog, "Success", f"Drone '{drone.id}' silindi!")
                    dialog.accept()
                    
                except Exception as e:
                    QMessageBox.critical(dialog, "Error", f"Drone silinirken hata oluştu:\n{str(e)}")
        
        remove_btn.clicked.connect(remove_selected)
        cancel_btn.clicked.connect(dialog.reject)
        
        button_layout.addWidget(remove_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        dialog.setLayout(layout)
        dialog.exec_()
    
    def show_remove_delivery_dialog(self):
        """Delivery seçip silme dialogu"""
        if not self.system.deliveries:
            QMessageBox.information(self, "Info", "Silinecek delivery yok!")
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Remove Delivery")
        dialog.setModal(True)
        dialog.resize(500, 350)
        
        layout = QVBoxLayout()
        
        # Açıklama
        info_label = QLabel("Silmek istediğiniz delivery'yi seçin:")
        info_label.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        # Delivery listesi
        delivery_list = QTableWidget()
        delivery_list.setColumnCount(5)
        delivery_list.setHorizontalHeaderLabels(["ID", "Position", "Weight", "Priority", "Status"])
        delivery_list.setRowCount(len(self.system.deliveries))
        delivery_list.setSelectionBehavior(QTableWidget.SelectRows)
        
        for i, delivery in enumerate(self.system.deliveries):
            delivery_list.setItem(i, 0, QTableWidgetItem(delivery.id))
            delivery_list.setItem(i, 1, QTableWidgetItem(f"({delivery.position[0]:.1f}, {delivery.position[1]:.1f})"))
            delivery_list.setItem(i, 2, QTableWidgetItem(f"{delivery.weight:.1f}kg"))
            delivery_list.setItem(i, 3, QTableWidgetItem(str(delivery.priority)))
            status = getattr(delivery, 'status', 'pending')
            delivery_list.setItem(i, 4, QTableWidgetItem(status))
        
        layout.addWidget(delivery_list)
        
        # Butonlar - standart stil
        button_layout = QHBoxLayout()
        
        remove_btn = QPushButton("Remove Selected")
        cancel_btn = QPushButton("Cancel")
        
        def remove_selected():
            selected_rows = delivery_list.selectionModel().selectedRows()
            if not selected_rows:
                QMessageBox.warning(dialog, "Warning", "Lütfen silinecek delivery'yi seçin!")
                return
            
            row = selected_rows[0].row()
            delivery = self.system.deliveries[row]
            
            reply = QMessageBox.question(
                dialog,
                "Confirm Remove",
                f"'{delivery.id}' delivery'sini silmek istediğinizden emin misiniz?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                try:
                    # System'den sil
                    self.system.deliveries.remove(delivery)
                    
                    # Optimizer varsa oradan da sil
                    if hasattr(self.system, 'optimizer') and self.system.optimizer:
                        self.system.optimizer.remove_delivery(delivery.id)
                    
                    # Görselleştirmeyi güncelle
                    self.update_visualization()
                    
                    QMessageBox.information(dialog, "Success", f"Delivery '{delivery.id}' silindi!")
                    dialog.accept()
                    
                except Exception as e:
                    QMessageBox.critical(dialog, "Error", f"Delivery silinirken hata oluştu:\n{str(e)}")
        
        remove_btn.clicked.connect(remove_selected)
        cancel_btn.clicked.connect(dialog.reject)
        
        button_layout.addWidget(remove_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        dialog.setLayout(layout)
        dialog.exec_()
    
    def show_remove_zone_dialog(self):
        """No-fly zone seçip silme dialogu"""
        if not self.system.no_fly_zones:
            QMessageBox.information(self, "Info", "Silinecek no-fly zone yok!")
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Remove No-Fly Zone")
        dialog.setModal(True)
        dialog.resize(400, 300)
        
        layout = QVBoxLayout()
        
        # Açıklama
        info_label = QLabel("Silmek istediğiniz no-fly zone'u seçin:")
        info_label.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        # Zone listesi
        zone_list = QTableWidget()
        zone_list.setColumnCount(3)
        zone_list.setHorizontalHeaderLabels(["ID", "Coordinates", "Active Time"])
        zone_list.setRowCount(len(self.system.no_fly_zones))
        zone_list.setSelectionBehavior(QTableWidget.SelectRows)
        
        for i, zone in enumerate(self.system.no_fly_zones):
            zone_list.setItem(i, 0, QTableWidgetItem(getattr(zone, 'id', f"Zone-{i+1}")))
            
            # Koordinatları kısalt
            coords = str(zone.polygon_coordinates[:2]) + "..." if len(zone.polygon_coordinates) > 2 else str(zone.polygon_coordinates)
            zone_list.setItem(i, 1, QTableWidgetItem(coords))
            
            # Aktif zamanı - doğru attribute isimlerini kullan
            active_time = f"{zone.active_time_start.strftime('%H:%M')} - {zone.active_time_end.strftime('%H:%M')}"
            zone_list.setItem(i, 2, QTableWidgetItem(active_time))
        
        layout.addWidget(zone_list)
        
        # Butonlar - standart stil
        button_layout = QHBoxLayout()
        
        remove_btn = QPushButton("Remove Selected")
        cancel_btn = QPushButton("Cancel")
        
        def remove_selected():
            selected_rows = zone_list.selectionModel().selectedRows()
            if not selected_rows:
                QMessageBox.warning(dialog, "Warning", "Lütfen silinecek no-fly zone'u seçin!")
                return
            
            row = selected_rows[0].row()
            zone = self.system.no_fly_zones[row]
            zone_id = getattr(zone, 'id', f"Zone-{row+1}")
            
            reply = QMessageBox.question(
                dialog,
                "Confirm Remove",
                f"'{zone_id}' no-fly zone'unu silmek istediğinizden emin misiniz?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                try:
                    # System'den sil
                    self.system.no_fly_zones.remove(zone)
                    
                    # Optimizer varsa oradan da sil
                    if hasattr(self.system, 'optimizer') and self.system.optimizer:
                        if zone in self.system.optimizer.no_fly_zones:
                            self.system.optimizer.no_fly_zones.remove(zone)
                    
                    # Görselleştirmeyi güncelle
                    self.update_visualization()
                    
                    QMessageBox.information(dialog, "Success", f"No-fly zone '{zone_id}' silindi!")
                    dialog.accept()
                    
                except Exception as e:
                    QMessageBox.critical(dialog, "Error", f"No-fly zone silinirken hata oluştu:\n{str(e)}")
        
        remove_btn.clicked.connect(remove_selected)
        cancel_btn.clicked.connect(dialog.reject)
        
        button_layout.addWidget(remove_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        dialog.setLayout(layout)
        dialog.exec_()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()