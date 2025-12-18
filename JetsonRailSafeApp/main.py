import sys
import cv2
import time
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFrame
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, pyqtSlot

# Import backend processor
from backend.processor import RailSafeEngine

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_info_signal = pyqtSignal(dict)

    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self._run_flag = True
        self.paused = False

    def run(self):
        while self._run_flag:
            if not self.paused:
                frame, info = self.engine.process_next_frame()
                if frame is not None:
                    self.change_pixmap_signal.emit(frame)
                    self.update_info_signal.emit(info)
                else:
                    # End of video or error, maybe wait a bit or stop
                    time.sleep(0.1)
            else:
                time.sleep(0.1)

    def stop(self):
        self._run_flag = False
        self.wait()

    def toggle_pause(self):
        self.paused = not self.paused

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RailSafeNet - Jetson Orin Edition")
        self.setGeometry(100, 100, 1280, 720)
        self.setStyleSheet("background-color: #1e1e1e; color: white;")

        # Initialize Engine
        self.engine = RailSafeEngine()
        
        self.initUI()

        # Start Video Thread
        self.thread = VideoThread(self.engine)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_info_signal.connect(self.update_info)
        self.thread.start()

    def initUI(self):
        # Main Layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # Top Section (Video + Status)
        top_layout = QHBoxLayout()
        
        # Video Feed (Left)
        self.image_label = QLabel(self)
        self.image_label.resize(896, 512)
        self.image_label.setStyleSheet("background-color: black; border: 2px solid #333;")
        self.image_label.setAlignment(Qt.AlignCenter)
        top_layout.addWidget(self.image_label, stretch=3)

        # Status Panel (Right)
        status_panel = QFrame()
        status_panel.setStyleSheet("background-color: #2d2d2d; border-radius: 10px; padding: 10px;")
        status_layout = QVBoxLayout()
        status_panel.setLayout(status_layout)

        # Status Indicator
        self.status_label = QLabel("STATUS")
        self.status_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.status_label)

        self.status_indicator = QLabel("SAFE")
        self.status_indicator.setFont(QFont("Arial", 24, QFont.Bold))
        self.status_indicator.setAlignment(Qt.AlignCenter)
        self.status_indicator.setStyleSheet("background-color: green; color: white; border-radius: 5px; padding: 10px;")
        self.status_indicator.setFixedHeight(100)
        status_layout.addWidget(self.status_indicator)

        # Stats
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setFont(QFont("Arial", 12))
        status_layout.addWidget(self.fps_label)
        
        self.cache_label = QLabel("Cache: Init")
        self.cache_label.setFont(QFont("Arial", 10))
        self.cache_label.setWordWrap(True)
        status_layout.addWidget(self.cache_label)

        status_layout.addStretch()
        top_layout.addWidget(status_panel, stretch=1)

        main_layout.addLayout(top_layout, stretch=4)

        # Bottom Section (Speed + Controls)
        bottom_layout = QHBoxLayout()

        # Speed Gauge (Left)
        speed_panel = QFrame()
        speed_panel.setStyleSheet("background-color: #2d2d2d; border-radius: 10px; padding: 10px;")
        speed_layout = QVBoxLayout()
        speed_panel.setLayout(speed_layout)
        
        self.speed_label = QLabel("SPEED")
        self.speed_label.setAlignment(Qt.AlignCenter)
        speed_layout.addWidget(self.speed_label)
        
        self.speed_value = QLabel("0 km/h")
        self.speed_value.setFont(QFont("Arial", 20, QFont.Bold))
        self.speed_value.setAlignment(Qt.AlignCenter)
        self.speed_value.setStyleSheet("color: #00ccff;")
        speed_layout.addWidget(self.speed_value)
        
        bottom_layout.addWidget(speed_panel, stretch=1)

        # Controls (Right)
        controls_panel = QFrame()
        controls_layout = QHBoxLayout()
        controls_panel.setLayout(controls_layout)

        self.btn_pause = QPushButton("PAUSE")
        self.btn_pause.setStyleSheet("background-color: #ff9900; color: black; font-weight: bold; padding: 10px;")
        self.btn_pause.clicked.connect(self.toggle_pause)
        controls_layout.addWidget(self.btn_pause)

        self.btn_quit = QPushButton("QUIT")
        self.btn_quit.setStyleSheet("background-color: #ff3333; color: white; font-weight: bold; padding: 10px;")
        self.btn_quit.clicked.connect(self.close_app)
        controls_layout.addWidget(self.btn_quit)

        bottom_layout.addWidget(controls_panel, stretch=3)
        main_layout.addLayout(bottom_layout, stretch=1)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    @pyqtSlot(dict)
    def update_info(self, info):
        """Updates status labels"""
        # FPS
        self.fps_label.setText(f"FPS: {info.get('fps', 0):.1f}")
        
        # Cache Stats
        self.cache_label.setText(f"Seg Cache: {info.get('seg_cache', 0):.0f}%\nDet Cache: {info.get('det_cache', 0):.0f}%")

        # Status Color
        status = info.get('status', 'SAFE')
        color = "green"
        if status == 'DANGER': color = "red"
        elif status == 'WARNING': color = "orange"
        elif status == 'CAUTION': color = "yellow"
        
        self.status_indicator.setText(status)
        self.status_indicator.setStyleSheet(f"background-color: {color}; color: {'black' if color in ['yellow', 'orange'] else 'white'}; border-radius: 5px; padding: 10px;")

        # Speed (Mockup for now)
        # self.speed_value.setText(f"{info.get('speed', 0)} km/h")

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def toggle_pause(self):
        self.thread.toggle_pause()
        if self.thread.paused:
            self.btn_pause.setText("RESUME")
            self.btn_pause.setStyleSheet("background-color: #00cc00; color: white; font-weight: bold; padding: 10px;")
        else:
            self.btn_pause.setText("PAUSE")
            self.btn_pause.setStyleSheet("background-color: #ff9900; color: black; font-weight: bold; padding: 10px;")

    def close_app(self):
        self.thread.stop()
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
