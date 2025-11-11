import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def preprocess_image(image: np.ndarray) -> np.ndarray:

    # 1. Convertește imaginea în grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Aplică un filtru Gaussian pentru a reduce zgomotul
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Aplică detecția de contururi Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    return edges