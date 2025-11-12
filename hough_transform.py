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


# if __name__ == "__main__":
#     while True:
#         image_path = input("Introdu calea catre imaginea dorita sau 'exit' pentru a iesi: ")
#         if image_path.lower() == 'exit':
#             break
        
#         # Verificarea căii
#         if not os.path.isfile(image_path):
#             print(f"Eroare: Calea '{image_path}' nu este valida.")
#             continue
        
#         image = cv2.imread(image_path)
#         processed_edges = preprocess_image(image)
        
#         accumulator, thetas_rad, rhos = hough_transform(processed_edges)
        
#         figure = plt.figure(figsize=(18, 6))
        
#         # 1. Imaginea Originală
#         plt.subplot(1, 3, 1)
#         plt.title("Imagine Originala")
        
#         plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         plt.axis('off')
        
#         # 2. Contururile Canny
#         plt.subplot(1, 3, 2)
#         plt.title("Imagine Preprocesata (Contururi Canny)")
#         plt.imshow(processed_edges, cmap='gray')
#         plt.axis('off')
        
#         # 3. Spațiul Acumulatorului Hough
#         plt.subplot(1, 3, 3)
#         plt.title("Spatiul Acumulatorului Hough (Rho-Theta)")
        
#         plt.imshow(np.log1p(accumulator), 
#                    cmap='jet', 
#                    aspect='auto', 
#                    extent=[np.rad2deg(thetas_rad[0]), np.rad2deg(thetas_rad[-1]), rhos[-1], rhos[0]]) 
#         plt.xlabel("Theta (Grade)")
#         plt.ylabel("Rho (Pixeli)")
        
#         plt.tight_layout()

#         plt.show()
