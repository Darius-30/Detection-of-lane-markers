import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Pasul 1: Preprocesarea imaginii (Canny).
    (Acesta este codul tău, aprox. 15%)
    """
    # Convertește imaginea în grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplică un filtru Gaussian pentru a reduce zgomotul
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Aplică detecția de contururi Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    return edges

# --- SECȚIUNE NOUĂ (Următorii 40%) ---

def custom_hough_transform(image: np.ndarray, theta_resolution_deg: int = 1, rho_resolution_pix: int = 1) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Pasul 2: Implementarea manuală a Transformatei Hough.
    Construiește spațiul acumulatorului (Rho-Theta).
    Aceasta este funcția care implementează ecuația (1) din articol.
    """
    
    # --- 1. Definirea Spațiului Acumulatorului ---
    
    # Aflăm dimensiunile imaginii
    height, width = image.shape
    
    # Diagonala imaginii este valoarea maximă posibilă pentru Rho (ρ)
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
    
    # --- 2. Procesul de Votare ---
    
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
        
        # Aplicăm ecuația (1) pentru TOATE unghiurile theta simultan
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

# --- Sfârșitul secțiunii noi ---


# --- Blocul principal (modificat pentru a include Pasul 2) ---

if __name__ == "__main__":
    while True:
        image_path = input("Introdu calea catre imaginea dorita sau 'exit' pentru a iesi: ")
        if image_path.lower() == 'exit':
            break
        
        # Verificarea căii (din codul tău)
        if not os.path.isfile(image_path):
            print(f"Eroare: Calea '{image_path}' nu este valida.")
            continue
            
        # --- PASUL 1: PREPROCESARE (din codul tău) ---
        image = cv2.imread(image_path)
        processed_edges = preprocess_image(image)
        
        # --- PASUL 2: CONSTRUIREA ACUMULATORULUI (NOU) ---
        # Aici apelăm noua funcție
        accumulator, thetas_rad, rhos = custom_hough_transform(processed_edges)
        
        
        # --- VIZUALIZARE (Extinsă la 3 grafice) ---
        
        # Am schimbat la 1 rând, 3 coloane (1, 3) și figsize
        figure = plt.figure(figsize=(18, 6))
        
        # 1. Imaginea Originală
        plt.subplot(1, 3, 1)
        plt.title("Imagine Originala")
        # Corecție BGR -> RGB pentru afișare corectă în Matplotlib
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        # 2. Contururile Canny
        plt.subplot(1, 3, 2)
        plt.title("Imagine Preprocesata (Contururi Canny)")
        plt.imshow(processed_edges, cmap='gray')
        plt.axis('off')
        
        # 3. Spațiul Acumulatorului Hough (NOU)
        plt.subplot(1, 3, 3)
        plt.title("Spatiul Acumulatorului Hough (Rho-Theta)")
        
        # Folosim np.log1p pentru a vizualiza mai bine vârfurile
        # (vârfurile au valori f. mari, ex. 500, iar restul au 0-1)
        # Logaritmul compresează intervalul și face "fluturii" vizibili
        
        plt.imshow(np.log1p(accumulator), 
                   cmap='jet', 
                   aspect='auto', 
                   extent=[np.rad2deg(thetas_rad[0]), np.rad2deg(thetas_rad[-1]), rhos[-1], rhos[0]]) # Am inversat rhos[0] și rhos[-1] pentru o afișare corectă
        plt.xlabel("Theta (Grade)")
        plt.ylabel("Rho (Pixeli)")
        
        plt.tight_layout() # Aranjează graficele frumos
        plt.show()