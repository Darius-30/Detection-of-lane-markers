import cv2
import numpy as np
import matplotlib.pyplot as plt
import os    # Biblioteca pentru a lucra cu căi de fișiere

def preprocess_image(image: np.ndarray) -> np.ndarray:

    # 2. Convertește imaginea în grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 3. Aplică un filtru Gaussian pentru a reduce zgomotul
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 4. Aplică detecția de contururi Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    return edges

# --- Exemplu de utilizare cu dataset-ul TuSimple ---
if __name__ == "__main__":
    while True:
        image_path = input("Introdu calea catre imaginea dorita sau 'exit' pentru a iesi: ")
        if image_path.lower() == 'exit':
            break
        if not os.path.isfile(image_path):
            print(f"Eroare: Calea '{image_path}' nu este valida.")
            continue
        image = cv2.imread(image_path)
        processed_edges = preprocess_image(image)
        
        figure = plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.title("Imagine Originala (din TuSimple)")
        plt.imshow(image)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title("Imagine Preprocesata (Contururi Canny)")
        plt.imshow(processed_edges, cmap='gray')
        plt.axis('off')
        
        plt.show()
