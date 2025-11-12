import sys
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, 
    QPushButton, QLabel, QFileDialog, QStackedWidget
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


from hough_transform import hough_transform
from preprocess_image import preprocess_image



class ImageProcessorApp(QWidget):
    def __init__(self):
        super().__init__()
        
        self.original_cv_image = None
        self.processed_cv_image = None 
        
        self.hough_figure = Figure(figsize=(14, 10), dpi=100)
        self.hough_canvas = FigureCanvas(self.hough_figure)
        
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Procesor de Imagini - Transformata Hough cu Axe')
        self.setGeometry(100, 100, 1400, 700) 

        main_layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        
        self.label_original_title = QLabel('Imagine Originală')
        self.label_original_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.image_label_original = QLabel()
        self.image_label_original.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label_original.setFixedSize(700, 400)
        self.image_label_original.setStyleSheet("border: 1px solid gray;")
        self.image_label_original.setText("Încărcați o imagine...")
        
        self.label_processed_title = QLabel('Imagine Procesată')
        self.label_processed_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.processed_stack = QStackedWidget()
        self.processed_stack.setFixedSize(700, 400)
        
        self.image_label_processed = QLabel()
        self.image_label_processed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label_processed.setStyleSheet("border: 1px solid gray;")
        self.image_label_processed.setText("Așteptare procesare...")
        

        self.processed_stack.addWidget(self.image_label_processed)
        self.processed_stack.addWidget(self.hough_canvas)

        left_layout.addWidget(self.label_original_title)
        left_layout.addWidget(self.image_label_original)
        left_layout.addSpacing(10)
        left_layout.addWidget(self.label_processed_title)
        left_layout.addWidget(self.processed_stack) 
        left_layout.addStretch()

        right_layout = QVBoxLayout()
        
        self.btn_load = QPushButton('Încarcă Imagine')
        self.btn_preprocess = QPushButton('1. Preprocesare (Canny)')
        self.btn_hough = QPushButton('2. Aplică Transformata Hough')
        
        self.btn_preprocess.setEnabled(False)
        self.btn_hough.setEnabled(False)

        self.btn_load.clicked.connect(self.load_image)
        self.btn_preprocess.clicked.connect(self.run_preprocess)
        self.btn_hough.clicked.connect(self.run_hough)

        right_layout.addWidget(self.btn_load)
        right_layout.addWidget(self.btn_preprocess)
        right_layout.addWidget(self.btn_hough)
        right_layout.addStretch() 

        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 1)
        
        self.setLayout(main_layout)

    def display_pixmap_in_label(self, label: QLabel, pixmap: QPixmap):
        scaled_pixmap = pixmap.scaled(
            label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Deschide Imagine', '', 'Fișiere Imagine (*.png *.jpg *.jpeg *.bmp)')
        
        if fname:
            pixmap = QPixmap(fname)
            scaled_pixmap = pixmap.scaled(
                self.image_label_original.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label_original.setPixmap(scaled_pixmap)
            
            self.original_cv_image = qpixmap_to_ndarray(pixmap) 
            
            self.image_label_processed.clear()
            self.image_label_processed.setText("Așteptare procesare...")
            self.label_processed_title.setText('Imagine Procesată')
            self.processed_cv_image = None
            
            self.processed_stack.setCurrentWidget(self.image_label_processed) 
            
            self.btn_preprocess.setEnabled(True)
            self.btn_hough.setEnabled(False)

    def run_preprocess(self):
        if self.original_cv_image is not None:
            self.processed_cv_image = preprocess_image(self.original_cv_image)
            pixmap_result = ndarray_to_qpixmap(self.processed_cv_image)
            
            self.display_pixmap_in_label(self.image_label_processed, pixmap_result)
            self.label_processed_title.setText('Imagine Procesată (Contururi Canny)')
            
            self.processed_stack.setCurrentWidget(self.image_label_processed)
            
            self.btn_hough.setEnabled(True)
        else:
            print("Nu există o imagine originală încărcată.")

    def run_hough(self):
        if self.processed_cv_image is not None:
            accumulator, thetas_rad, rhos = hough_transform(self.processed_cv_image)

            self.hough_figure.clear()
            
            ax = self.hough_figure.add_subplot(111)
            
            log_accumulator = np.log1p(accumulator)
            
            extent = [
                np.rad2deg(thetas_rad[0]), 
                np.rad2deg(thetas_rad[-1]), 
                rhos[-1], 
                rhos[0]
            ]
            
            ax.imshow(
                log_accumulator, 
                cmap='jet', 
                aspect='auto', 
                extent=extent
            )
            
            ax.set_xlabel("Theta (Grade)")
            ax.set_ylabel("Rho (Pixeli)")
            
            self.hough_figure.tight_layout()
            
            self.hough_canvas.draw()
            
            self.label_processed_title.setText('Spațiul Acumulatorului Hough')
            
            self.processed_stack.setCurrentWidget(self.hough_canvas)
            
        else:
            print("Rulați mai întâi preprocesarea (Pasul 1).")

def qpixmap_to_ndarray(pixmap: QPixmap) -> np.ndarray:
        q_image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        width, height = q_image.width(), q_image.height()
        ptr = q_image.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4))
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

def ndarray_to_qpixmap(cv_image: np.ndarray) -> QPixmap:
    height, width = cv_image.shape[:2]
    if len(cv_image.shape) == 2:
        bytes_per_line = width
        q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
    else:
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        channels = rgb_image.shape[2]
        bytes_per_line = channels * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(q_image.copy())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageProcessorApp()
    window.show()
    sys.exit(app.exec())