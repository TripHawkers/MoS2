import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from ultralytics import YOLO
import cv2
import numpy as np
import mss
import torch

class ImageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.model = None
        self.image_path = ""
        self.weight_path = ""
        self.annotated_image = None

    def init_ui(self):
        self.setWindowTitle('IMAGE APP')
        self.setGeometry(600, 300, 300, 200)

        self.layout = QVBoxLayout()
        image_layout = QHBoxLayout()

        self.original_image_layout = QVBoxLayout()
        self.image_label = QLabel(self)
        self.original_image_layout.addWidget(self.image_label)

        self.title_original = QLabel("Original image", self)
        self.title_original.setAlignment(Qt.AlignCenter)
        self.title_original.setVisible(False)
        self.original_image_layout.addWidget(self.title_original)

        self.segmented_image_layout = QVBoxLayout()
        self.segmented_image_label = QLabel(self)
        self.segmented_image_layout.addWidget(self.segmented_image_label)

        self.title_segmented = QLabel("Processed image", self)
        self.title_segmented.setAlignment(Qt.AlignCenter)
        self.title_segmented.setVisible(False)
        self.segmented_image_layout.addWidget(self.title_segmented)

        image_layout.addLayout(self.original_image_layout)
        image_layout.addLayout(self.segmented_image_layout)

        self.layout.addLayout(image_layout)

        self.load_weights_button = QPushButton('Load weight', self)
        self.load_weights_button.clicked.connect(self.load_weights)
        self.layout.addWidget(self.load_weights_button)

        self.inference_button = QPushButton('Real-time inference', self)
        self.inference_button.clicked.connect(self.inference)
        self.layout.addWidget(self.inference_button)

        self.select_image_button = QPushButton('Select image', self)
        self.select_image_button.clicked.connect(self.select_image)
        self.layout.addWidget(self.select_image_button)

        self.detect_button = QPushButton('Inference', self)
        self.detect_button.clicked.connect(self.detect_objects)
        self.layout.addWidget(self.detect_button)

        self.save_button = QPushButton('Save result', self)
        self.save_button.clicked.connect(self.save_image)
        self.layout.addWidget(self.save_button)

        self.exit_button = QPushButton('Quit', self)
        self.exit_button.clicked.connect(self.close_application)
        self.layout.addWidget(self.exit_button)

        self.error_label = QLabel("", self)
        self.error_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.error_label)

        widget = QWidget(self)
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)

    def select_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select image', '', 'Images (*.png *.xpm *.jpg)')
        if file_name:
            self.image_path = file_name
            self.display_image(self.image_path, self.image_label)
            self.title_original.setText("Original image")
            self.title_original.setVisible(True)
            self.title_segmented.setVisible(False)
            self.error_label.setText("Image imported successfully")
            self.error_label.setStyleSheet("color: green;")

    def display_image(self, path, label):
        pixmap = QPixmap(path)
        pixmap = pixmap.scaled(600, 600, Qt.KeepAspectRatio)
        label.setPixmap(pixmap)

    def detect_objects(self):
        if not self.image_path:
            self.error_label.setText("No image loaded")
            self.error_label.setStyleSheet("color: red;")
            return
        if not self.weight_path:
            self.error_label.setText("No weights loaded")
            self.error_label.setStyleSheet("color: red;")
            return

        results = self.model(self.image_path)
        img = cv2.imread(self.image_path)
        annotated_image = results[0].plot()
        self.annotated_image = annotated_image
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        height, width, channel = annotated_image_rgb.shape
        bytes_per_line = 3 * width
        qimg = QImage(annotated_image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        qpixmap = QPixmap.fromImage(qimg).scaled(600, 600, Qt.KeepAspectRatio)
        self.segmented_image_label.setPixmap(qpixmap)

        self.title_segmented.setText("Inference result")
        self.title_segmented.setVisible(True)

        self.adjustSize()
        self.setMinimumSize(self.width(), self.height())

        self.error_label.setText("Inference successful")
        self.error_label.setStyleSheet("color: green;")

    def save_image(self):
        if self.annotated_image is not None:
            file_name, _ = QFileDialog.getSaveFileName(self, 'Save results', '', 'Images (*.png *.xpm *.jpg)')
            if file_name:
                cv2.imwrite(file_name, self.annotated_image)
                self.error_label.setText("Image saved successfully")
                self.error_label.setStyleSheet("color: green;")
        else:
            self.error_label.setText("No inference results to save")
            self.error_label.setStyleSheet("color: red;")

    def load_weights(self):
        weight_file, _ = QFileDialog.getOpenFileName(self, 'Select weight file', '', 'Weights Files (*.pt)')

        if weight_file:
            self.weight_path = weight_file
            self.model = YOLO(weight_file)
            self.error_label.setText("Weight imported successfully")
            self.error_label.setStyleSheet("color: green;")

    def inference(self):
        if not self.weight_path:
            self.error_label.setText("No weight file loaded")
            self.error_label.setStyleSheet("color: red;")
            return

        self.set_ui_enabled(False)

        self.inference_button.setText("Quit inference")
        self.inference_button.clicked.disconnect()
        self.inference_button.clicked.connect(self.stop_inference)


        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = YOLO(self.weight_path)
        model.to(device)

        monitor = {
            "top": 200,
            "left": 200,
            "width": 800,
            "height": 600

        }

        self.inference_running = True

        with mss.mss() as sct:
            while self.inference_running:

                screenshot = sct.grab(monitor)
                img = np.array(screenshot)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                results = model(img)
                segmented_image = results[0].plot()
                cv2.imshow('Real-Time Inference', segmented_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()

    def stop_inference(self):
        self.inference_running = False
        self.set_ui_enabled(True)
        self.inference_button.setText("Real-time inference")
        self.inference_button.clicked.disconnect()
        self.inference_button.clicked.connect(self.inference)

        self.error_label.setText("Quit inference")
        self.error_label.setStyleSheet("color: green;")

    def set_ui_enabled(self, enabled):
        self.select_image_button.setEnabled(enabled)
        self.detect_button.setEnabled(enabled)
        self.save_button.setEnabled(enabled)
        self.load_weights_button.setEnabled(enabled)


    def close_application(self):
        self.inference_running = False
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageApp()
    window.show()
    sys.exit(app.exec_())