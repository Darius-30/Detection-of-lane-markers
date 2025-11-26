import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from preprocess_image import preprocess_image


def hough_transform(image: np.ndarray, theta_resolution_deg: int = 1, rho_resolution_pix: int = 1) -> (np.ndarray, np.ndarray, np.ndarray):

    height, width = image.shape
    
    max_rho = int(np.ceil(np.sqrt(height**2 + width**2)))
    
    # Definim axa Rho: de la -max_rho la +max_rho
    rho_bins_count = int(2 * max_rho / rho_resolution_pix)
    rhos = np.linspace(-max_rho, max_rho, rho_bins_count)
    
    # Definim axa Theta (θ): de la 0 la 180 grade
    theta_bins_count = int(180 / theta_resolution_deg)
    # Vectorul valorilor theta (în RADIANI, necesari pentru cos/sin)
    thetas = np.deg2rad(np.arange(0, 180, theta_resolution_deg))
    
    # Inițializăm acumulatorul cu zerouri
    accumulator = np.zeros((rho_bins_count, theta_bins_count), dtype=np.uint64)
    
    # 2. Procesul de Votare 
    
    # Găsim coordonatele (x, y) ale tuturor pixelilor de contur (albi)
    y_indices, x_indices = np.nonzero(image)
    
    # Pre-calculăm valorile cos și sin pentru toate unghiurile theta
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)

    print(f"Construirea acumulatorului... Se votează cu {len(x_indices)} puncte de contur.")
    
    # Iterăm prin fiecare punct de contur (x, y)
    for i in range(len(x_indices)):
        x = x_indices[i]
        y = y_indices[i]
        
        # Aplicăm ecuația (1) pentru toate unghiurile theta simultan
        calculated_rhos = x * cos_thetas + y * sin_thetas
        
        # Convertim valorile 'rho' (continue) în indici 'rho' (discreți)
        rho_indices = ((calculated_rhos + max_rho) / rho_resolution_pix).astype(int)
        
        # Iterăm prin fiecare unghi și votăm
        for theta_idx in range(theta_bins_count):
            rho_idx = rho_indices[theta_idx]
            if 0 <= rho_idx < rho_bins_count:
                accumulator[rho_idx, theta_idx] += 1
                
    print("Construirea acumulatorului a fost finalizată.")
    
    # Returnăm acumulatorul și axele corespunzătoare
    return accumulator, thetas, rhos
