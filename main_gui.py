import sys
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, 
    QPushButton, QLabel, QFileDialog, QStackedWidget, QMessageBox,
    QGroupBox, QSlider, QSpinBox, QFormLayout
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import module
from preprocess_image import preprocess_image
from hough_transform import hough_transform
from peak_detection import find_peaks
from line_reconstruction import solve_line_endpoints

class ImageProcessorApp(QWidget):
    def __init__(self):
        super().__init__()
        
        self.original_cv_image = None
        self.processed_cv_image = None
        self.hough_accumulator = None
        self.hough_thetas = None
        self.hough_rhos = None
        self.final_result_image = None
        
        self.hough_figure = Figure(figsize=(10, 8), dpi=100)
        self.hough_canvas = FigureCanvas(self.hough_figure)
        
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Robust Hough Transform - Laborator Tuning')
        self.setGeometry(50, 50, 1600, 900) 

        main_layout = QHBoxLayout()

        # === ZONA STÂNGA: Vizualizare ===
        left_layout = QVBoxLayout()
        
        self.label_original_title = QLabel('Imagine Originală')
        self.label_original_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label_original = QLabel("1. Încărcați o imagine...")
        self.image_label_original.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label_original.setFixedSize(700, 400)
        self.image_label_original.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        
        self.label_processed_title = QLabel('Rezultat Procesare')
        self.label_processed_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.processed_stack = QStackedWidget()
        self.processed_stack.setFixedSize(700, 400)
        
        self.image_label_processed = QLabel("Așteptare procesare...")
        self.image_label_processed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label_processed.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        
        self.processed_stack.addWidget(self.image_label_processed) # 0
        self.processed_stack.addWidget(self.hough_canvas)          # 1

        left_layout.addWidget(self.label_original_title)
        left_layout.addWidget(self.image_label_original)
        left_layout.addSpacing(10)
        left_layout.addWidget(self.label_processed_title)
        left_layout.addWidget(self.processed_stack) 
        left_layout.addStretch()

        # === ZONA DREAPTA: Controale ===
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)
        
        # 1. Buton Load
        self.btn_load = QPushButton('📂 Încarcă Imagine')
        self.btn_load.setStyleSheet("padding: 10px; font-weight: bold; font-size: 14px;")
        self.btn_load.clicked.connect(self.load_image)
        right_layout.addWidget(self.btn_load)

        # --- GRUP 1: Preprocesare (Canny) ---
        group_canny = QGroupBox("1. Parametri Preprocesare (Canny)")
        layout_canny = QFormLayout()
        
        # Slider Canny Low
        self.slider_canny_low = QSlider(Qt.Orientation.Horizontal)
        self.slider_canny_low.setRange(0, 255)
        self.slider_canny_low.setValue(50)
        self.lbl_canny_low = QLabel("50")
        self.slider_canny_low.valueChanged.connect(lambda v: self.lbl_canny_low.setText(str(v)))
        layout_canny.addRow("Prag Minim:", self.slider_canny_low)
        layout_canny.addRow("", self.lbl_canny_low)

        # Slider Canny High
        self.slider_canny_high = QSlider(Qt.Orientation.Horizontal)
        self.slider_canny_high.setRange(0, 255)
        self.slider_canny_high.setValue(150)
        self.lbl_canny_high = QLabel("150")
        self.slider_canny_high.valueChanged.connect(lambda v: self.lbl_canny_high.setText(str(v)))
        layout_canny.addRow("Prag Maxim:", self.slider_canny_high)
        layout_canny.addRow("", self.lbl_canny_high)
        
        self.btn_preprocess = QPushButton('Aplică Canny')
        self.btn_preprocess.clicked.connect(self.run_preprocess)
        layout_canny.addRow(self.btn_preprocess)
        
        group_canny.setLayout(layout_canny)
        right_layout.addWidget(group_canny)

        # --- GRUP 2: Hough & Peak Detection ---
        group_hough = QGroupBox("2. Detecție Vârfuri (Peak Detection)")
        layout_hough = QFormLayout()

        self.btn_hough_calc = QPushButton('Calculează Hough (Matrice)')
        self.btn_hough_calc.clicked.connect(self.run_hough_transform)
        layout_hough.addRow(self.btn_hough_calc)
        
        # Neighborhood Size
        self.spin_neighborhood = QSpinBox()
        self.spin_neighborhood.setRange(5, 100)
        self.spin_neighborhood.setValue(30)
        layout_hough.addRow("Neighborhood Size (px):", self.spin_neighborhood)
        
        # Threshold Relativ (Slider 0-100 -> float 0.0-1.0)
        self.slider_thresh_rel = QSlider(Qt.Orientation.Horizontal)
        self.slider_thresh_rel.setRange(1, 100)
        self.slider_thresh_rel.setValue(40) # 0.40
        self.lbl_thresh_rel = QLabel("0.40")
        self.slider_thresh_rel.valueChanged.connect(lambda v: self.lbl_thresh_rel.setText(f"{v/100:.2f}"))
        layout_hough.addRow("Prag Relativ Intensitate:", self.slider_thresh_rel)
        layout_hough.addRow("", self.lbl_thresh_rel)
        
        self.btn_show_hough = QPushButton('Afișează Spațiul Hough')
        self.btn_show_hough.clicked.connect(self.show_hough_space)
        layout_hough.addRow(self.btn_show_hough)

        group_hough.setLayout(layout_hough)
        right_layout.addWidget(group_hough)

        # --- GRUP 3: Reconstrucție ---
        group_rec = QGroupBox("3. Reconstrucție Segmente (Butterfly)")
        layout_rec = QFormLayout()
        
        # Delta Deg
        self.spin_delta = QSpinBox()
        self.spin_delta.setRange(1, 20)
        self.spin_delta.setValue(3)
        layout_rec.addRow("Delta Theta (Grade):", self.spin_delta)

        # Scan Threshold (Lungime Linie)
        self.slider_scan_thresh = QSlider(Qt.Orientation.Horizontal)
        self.slider_scan_thresh.setRange(10, 95)
        self.slider_scan_thresh.setValue(70) # 0.70
        self.lbl_scan_thresh = QLabel("0.70 (Strict)")
        self.slider_scan_thresh.valueChanged.connect(lambda v: self.lbl_scan_thresh.setText(f"{v/100:.2f}"))
        layout_rec.addRow("Prag Scanare Lungime:", self.slider_scan_thresh)
        layout_rec.addRow("", self.lbl_scan_thresh)
        
        self.btn_reconstruct = QPushButton('GENEREAZĂ LINIILE')
        self.btn_reconstruct.setStyleSheet("background-color: #d4edda; font-weight: bold; padding: 8px;")
        self.btn_reconstruct.clicked.connect(self.run_reconstruction)
        layout_rec.addRow(self.btn_reconstruct)
        
        group_rec.setLayout(layout_rec)
        right_layout.addWidget(group_rec)

        right_layout.addStretch()

        # Add to Main
        main_layout.addLayout(left_layout, 3)
        main_layout.addLayout(right_layout, 1)
        self.setLayout(main_layout)
        
        # State
        self.enable_controls(False)

    # --- ACEASTA ESTE FUNCȚIA CARE LIPSEA ---
    def display_pixmap_in_label(self, label: QLabel, pixmap: QPixmap):
        scaled_pixmap = pixmap.scaled(
            label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)
    # ---------------------------------------

    def enable_controls(self, state):
        self.btn_preprocess.setEnabled(state)
        self.btn_hough_calc.setEnabled(state)
        self.btn_show_hough.setEnabled(state)
        self.btn_reconstruct.setEnabled(state)

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Deschide Imagine', '', 'Images (*.png *.jpg *.jpeg *.bmp)')
        if fname:
            pixmap = QPixmap(fname)
            self.display_pixmap_in_label(self.image_label_original, pixmap)
            self.original_cv_image = qpixmap_to_ndarray(pixmap)
            
            # Reset
            self.processed_cv_image = None
            self.hough_accumulator = None
            self.processed_stack.setCurrentIndex(0)
            self.image_label_processed.setText("Apasă 'Aplică Canny'...")
            
            self.enable_controls(True)
            self.btn_hough_calc.setEnabled(False)
            self.btn_reconstruct.setEnabled(False)

    def run_preprocess(self):
        if self.original_cv_image is None: return
        
        # Luăm valorile din Slidere
        low_thresh = self.slider_canny_low.value()
        high_thresh = self.slider_canny_high.value()
        
        # Reimplementăm logica preprocess aici pentru a folosi parametrii dinamici
        # sau modificăm preprocess_image.py. Pentru rapiditate, copiez logica aici:
        from preprocess_image import mask_region_of_interest # Păstrăm ROI
        
        gray = cv2.cvtColor(self.original_cv_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, low_thresh, high_thresh)
        final_edges = mask_region_of_interest(edges)
        
        self.processed_cv_image = final_edges
        
        pixmap_result = ndarray_to_qpixmap(self.processed_cv_image)
        self.display_pixmap_in_label(self.image_label_processed, pixmap_result)
        self.label_processed_title.setText('1. Canny Result')
        self.processed_stack.setCurrentIndex(0)
        
        self.btn_hough_calc.setEnabled(True)

    def run_hough_transform(self):
        if self.processed_cv_image is None: return
        
        # Calculăm acumulatorul (durează puțin)
        self.hough_accumulator, self.hough_thetas, self.hough_rhos = hough_transform(
            self.processed_cv_image, theta_resolution_deg=1
        )
        self.show_hough_space()
        self.btn_reconstruct.setEnabled(True)
        QMessageBox.information(self, "Info", "Transformata Hough Calculată! Poți ajusta parametrii și apăsa 'Generează Liniile'.")

    def show_hough_space(self):
        if self.hough_accumulator is None: return
        
        self.hough_figure.clear()
        ax = self.hough_figure.add_subplot(111)
        log_acc = np.log1p(self.hough_accumulator)
        extent = [np.rad2deg(self.hough_thetas[0]), np.rad2deg(self.hough_thetas[-1]), self.hough_rhos[-1], self.hough_rhos[0]]
        
        im = ax.imshow(log_acc, cmap='jet', aspect='auto', extent=extent)
        self.hough_figure.colorbar(im, ax=ax)
        ax.set_title("Spațiul Hough")
        ax.set_xlabel("Theta")
        ax.set_ylabel("Rho")
        self.hough_figure.tight_layout()
        self.hough_canvas.draw()
        
        self.label_processed_title.setText('2. Hough Space')
        self.processed_stack.setCurrentIndex(1)

    def run_reconstruction(self):
        if self.hough_accumulator is None: return

        # 1. Luăm parametrii din GUI
        thresh_rel = self.slider_thresh_rel.value() / 100.0
        neigh_size = self.spin_neighborhood.value()
        delta = self.spin_delta.value()
        scan_thresh = self.slider_scan_thresh.value() / 100.0

        # 2. Găsim Vârfurile
        peaks = find_peaks(
            self.hough_accumulator, 
            self.hough_thetas,
            threshold_rel=thresh_rel, 
            neighborhood_size=neigh_size
        )

        result_img = self.original_cv_image.copy()
        count = 0
        
        for rho_idx, theta_idx in peaks:
            endpoints = solve_line_endpoints(
                rho_idx, theta_idx, 
                self.hough_accumulator, 
                self.hough_thetas, 
                self.hough_rhos,
                delta_deg=delta,
                scan_threshold=scan_thresh # Parametrul nou!
            )
            
            if endpoints:
                pt1, pt2 = endpoints
                # Desenare (Subțire, transparentă e greu în OpenCV pur, facem solid)
                cv2.line(result_img, pt1, pt2, (0, 0, 255), 2)
                cv2.circle(result_img, pt1, 3, (0, 255, 0), -1)
                cv2.circle(result_img, pt2, 3, (0, 255, 0), -1)
                count += 1
        
        self.final_result_image = result_img
        pixmap_final = ndarray_to_qpixmap(self.final_result_image)
        self.display_pixmap_in_label(self.image_label_processed, pixmap_final)
        
        self.label_processed_title.setText(f'Rezultat: {count} segmente (Thresh={thresh_rel}, Neigh={neigh_size})')
        self.processed_stack.setCurrentIndex(0)


# --- UTILITARE ---
def qpixmap_to_ndarray(pixmap: QPixmap) -> np.ndarray:
    if pixmap.isNull(): return None
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