import sys
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, 
    QPushButton, QLabel, QFileDialog, QStackedWidget, QMessageBox
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# IMPORTURILE MODULELOR TALE
from preprocess_image import preprocess_image
from hough_transform import hough_transform
from peak_detection import find_peaks
from line_reconstruction_V2 import solve_line_endpoints

class ImageProcessorApp(QWidget):
    def __init__(self):
        super().__init__()
        
        self.original_cv_image = None
        self.processed_cv_image = None  # Canny
        self.hough_accumulator = None
        self.hough_thetas = None
        self.hough_rhos = None
        self.final_result_image = None # Imaginea cu liniile desenate
        
        self.hough_figure = Figure(figsize=(14, 10), dpi=100)
        self.hough_canvas = FigureCanvas(self.hough_figure)
        
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Reproducere Articol: Robust Hough Transform')
        self.setGeometry(100, 100, 1500, 750) 

        main_layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        
        # Titluri
        self.label_original_title = QLabel('Imagine Originală')
        self.label_original_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Vizualizare imagine stânga
        self.image_label_original = QLabel()
        self.image_label_original.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label_original.setFixedSize(700, 400)
        self.image_label_original.setStyleSheet("border: 1px solid gray;")
        self.image_label_original.setText("1. Încărcați o imagine...")
        
        self.label_processed_title = QLabel('Rezultat Procesare')
        self.label_processed_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Stacked Widget pentru a schimba între Canny / Hough Plot / Rezultat Final
        self.processed_stack = QStackedWidget()
        self.processed_stack.setFixedSize(700, 400)
        
        # Pagina 1: Imagine simplă (pentru Canny sau Linii Finale)
        self.image_label_processed = QLabel()
        self.image_label_processed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label_processed.setStyleSheet("border: 1px solid gray;")
        self.image_label_processed.setText("Așteptare procesare...")
        
        # Pagina 2: Graficul Matplotlib (pentru Hough)
        # (self.hough_canvas este deja creat în init)

        self.processed_stack.addWidget(self.image_label_processed) # Index 0
        self.processed_stack.addWidget(self.hough_canvas)          # Index 1

        left_layout.addWidget(self.label_original_title)
        left_layout.addWidget(self.image_label_original)
        left_layout.addSpacing(10)
        left_layout.addWidget(self.label_processed_title)
        left_layout.addWidget(self.processed_stack) 
        left_layout.addStretch()

        # --- Panoul de Control (Dreapta) ---
        right_layout = QVBoxLayout()
        
        self.btn_load = QPushButton('Încarcă Imagine')
        self.btn_preprocess = QPushButton('1. Preprocesare (Canny)')
        self.btn_hough = QPushButton('2. Transformata Hough')
        self.btn_reconstruct = QPushButton('3. Reconstrucție Segmente')
        
        # Stilizare butoane
        btn_style = "padding: 10px; font-size: 14px;"
        self.btn_load.setStyleSheet(btn_style)
        self.btn_preprocess.setStyleSheet(btn_style)
        self.btn_hough.setStyleSheet(btn_style)
        self.btn_reconstruct.setStyleSheet(btn_style + "font-weight: bold; color: darkblue;")
        
        self.btn_preprocess.setEnabled(False)
        self.btn_hough.setEnabled(False)
        self.btn_reconstruct.setEnabled(False)

        self.btn_load.clicked.connect(self.load_image)
        self.btn_preprocess.clicked.connect(self.run_preprocess)
        self.btn_hough.clicked.connect(self.run_hough)
        self.btn_reconstruct.clicked.connect(self.run_reconstruction)

        right_layout.addWidget(self.btn_load)
        right_layout.addSpacing(20)
        right_layout.addWidget(self.btn_preprocess)
        right_layout.addWidget(self.btn_hough)
        right_layout.addWidget(self.btn_reconstruct)
        right_layout.addStretch() 
        
        # Instrucțiuni
        info_label = QLabel("Instrucțiuni:\n1. Încarcă o imagine cu linii.\n2. Aplică Canny.\n3. Generează Acumulatorul.\n4. Detectează segmentele folosind metoda 'Butterfly'.")
        info_label.setWordWrap(True)
        right_layout.addWidget(info_label)

        main_layout.addLayout(left_layout, 3)
        main_layout.addLayout(right_layout, 1)
        
        self.setLayout(main_layout)

    # --- Helpers pentru imagini ---
    def display_pixmap_in_label(self, label: QLabel, pixmap: QPixmap):
        scaled_pixmap = pixmap.scaled(
            label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)

    # --- LOGICA BUTOANELOR ---

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Deschide Imagine', '', 'Fișiere Imagine (*.png *.jpg *.jpeg *.bmp)')
        
        if fname:
            pixmap = QPixmap(fname)
            self.display_pixmap_in_label(self.image_label_original, pixmap)
            
            self.original_cv_image = qpixmap_to_ndarray(pixmap) 
            
            # Resetare stare
            self.image_label_processed.clear()
            self.image_label_processed.setText("Așteptare procesare...")
            self.label_processed_title.setText('Rezultat Procesare')
            self.processed_cv_image = None
            self.hough_accumulator = None
            
            self.processed_stack.setCurrentIndex(0) # Show Label
            
            self.btn_preprocess.setEnabled(True)
            self.btn_hough.setEnabled(False)
            self.btn_reconstruct.setEnabled(False)

    def run_preprocess(self):
        if self.original_cv_image is not None:
            # APEL FUNCTIE EXTERNA
            self.processed_cv_image = preprocess_image(self.original_cv_image)
            
            pixmap_result = ndarray_to_qpixmap(self.processed_cv_image)
            self.display_pixmap_in_label(self.image_label_processed, pixmap_result)
            
            self.label_processed_title.setText('1. Contururi Canny Detectate')
            self.processed_stack.setCurrentIndex(0) # Show Label
            
            self.btn_hough.setEnabled(True)

    def run_hough(self):
        if self.processed_cv_image is not None:
            # APEL FUNCTIE EXTERNA
            # Poți ajusta theta_resolution_deg la 0.5 sau 1
            self.hough_accumulator, self.hough_thetas, self.hough_rhos = hough_transform(
                self.processed_cv_image, theta_resolution_deg=1
            )

            self.hough_figure.clear()
            ax = self.hough_figure.add_subplot(111)
            
            log_accumulator = np.log1p(self.hough_accumulator)
            
            extent = [
                np.rad2deg(self.hough_thetas[0]), 
                np.rad2deg(self.hough_thetas[-1]), 
                self.hough_rhos[-1], 
                self.hough_rhos[0]
            ]
            
            im = ax.imshow(log_accumulator, cmap='jet', aspect='auto', extent=extent)
            self.hough_figure.colorbar(im, ax=ax, label='Log(Voturi)')
            
            ax.set_title("2. Spațiul Acumulatorului (Rho-Theta)")
            ax.set_xlabel("Theta (Grade)")
            ax.set_ylabel("Rho (Pixeli)")
            
            self.hough_figure.tight_layout()
            self.hough_canvas.draw()
            
            self.label_processed_title.setText('2. Spațiul Hough')
            self.processed_stack.setCurrentIndex(1) # Show Canvas
            
            self.btn_reconstruct.setEnabled(True)

    def run_reconstruction(self):
        """
        Pasul final: Detecția vârfurilor și reconstrucția segmentelor
        """
        if self.hough_accumulator is None:
            return

        # 1. Găsim vârfurile (Peaks)
        # threshold_rel=0.3 înseamnă că luăm doar vârfurile care au 30% din intensitatea maximă
        peaks = find_peaks(self.hough_accumulator, self.hough_thetas, threshold_rel=0.30, neighborhood_size=30)
        
        if not peaks:
            QMessageBox.warning(self, "Atenție", "Nu au fost detectate vârfuri suficiente în spațiul Hough.")
            return

        print(f"S-au detectat {len(peaks)} vârfuri.")

        # 2. Creăm o copie a imaginii originale pentru a desena pe ea
        result_img = self.original_cv_image.copy()

        lines_found = 0
        for rho_idx, theta_idx in peaks:
            # 3. Calculăm punctele de capăt pentru fiecare vârf
            endpoints = solve_line_endpoints(
                rho_idx, theta_idx, 
                self.hough_accumulator, 
                self.hough_thetas, 
                self.hough_rhos,
                delta_deg=3 # Deschidere fereastră +/- 4 grade (ajustabil)
            )
            
            if endpoints:
                pt1, pt2 = endpoints
                # Desenăm linia pe imagine (Albastru gros)
                cv2.line(result_img, pt1, pt2, (0, 0, 255), 3)
                # Desenăm punctele de capăt (Cercuri verzi)
                cv2.circle(result_img, pt1, 5, (0, 255, 0), -1)
                cv2.circle(result_img, pt2, 5, (0, 255, 0), -1)
                lines_found += 1

        # 4. Afișăm rezultatul
        self.final_result_image = result_img
        pixmap_final = ndarray_to_qpixmap(self.final_result_image)
        
        self.display_pixmap_in_label(self.image_label_processed, pixmap_final)
        self.label_processed_title.setText(f'3. Rezultat Final: {lines_found} segmente detectate')
        self.processed_stack.setCurrentIndex(0) # Switch back to image view
        
        QMessageBox.information(self, "Succes", f"Proces complet! Au fost reconstruite {lines_found} segmente de linie.")

# --- Funcții Utilitare (Conversie Imagine) ---
def qpixmap_to_ndarray(pixmap: QPixmap) -> np.ndarray:
    if pixmap.isNull():
        return None
    q_image = pixmap.toImage()
    q_image = q_image.convertToFormat(QImage.Format.Format_RGBA8888)
    width, height = q_image.width(), q_image.height()
    ptr = q_image.bits()
    ptr.setsize(height * width * 4)
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4))
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

def ndarray_to_qpixmap(cv_image: np.ndarray) -> QPixmap:
    height, width = cv_image.shape[:2]
    if len(cv_image.shape) == 2: # Grayscale
        bytes_per_line = width
        q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
    else: # Color BGR
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