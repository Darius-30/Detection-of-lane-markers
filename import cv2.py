import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob  # Biblioteca pentru a găsi fișiere
import os    # Biblioteca pentru a lucra cu căi de fișiere

def preprocess_image_for_hough(image_path: str) -> np.ndarray:
    
    # 1. Încarcă imaginea
    image = cv2.imread(image_path)
    if image is None:
        print(f"Eroare: Nu am putut încărca imaginea de la: {image_path}")
        return None
    
    # 2. Convertește imaginea în grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 3. Aplică un filtru Gaussian pentru a reduce zgomotul
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 4. Aplică detecția de contururi Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    return edges

# --- Exemplu de utilizare cu dataset-ul TuSimple ---
if __name__ == "__main__":
    
    DATASET_CLIPS_PATH = r"D:\dataset PNI\TUSimple\train_set"

    search_path = os.path.join(DATASET_CLIPS_PATH, '**', '*.jpg')
    all_image_paths = glob.glob(search_path, recursive=True)
    
    if not all_image_paths:
        print(f"Eroare: Nu am găsit nicio imagine .jpg în calea: {DATASET_CLIPS_PATH}")
        print("Verifică dacă ai setat corect calea și dacă ai dezarhivat dataset-ul.")
    else:
        # Pentru început, hai să procesăm doar prima imagine găsită
        image_to_process = all_image_paths[0]
        
        print(f"Se procesează imaginea: {image_to_process}")
        
        # Rulează funcția de preprocesare
        processed_edges = preprocess_image_for_hough(image_to_process)
        
        if processed_edges is not None:
            # Afișează imaginea originală și imaginea procesată
            original_image = cv2.imread(image_to_process)
            original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.title("Imagine Originală (din TuSimple)")
            plt.imshow(original_image_rgb)
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.title("Imagine Preprocesată (Contururi Canny)")
            plt.imshow(processed_edges, cmap='gray')
            plt.axis('off')
            
            plt.show()