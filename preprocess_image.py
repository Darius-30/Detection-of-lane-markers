import cv2
import numpy as np

def mask_region_of_interest(edges_image: np.ndarray) -> np.ndarray:
    """
    Aplică o mască neagră peste zonele irelevante (cer, copaci),
    păstrând doar zona drumului (un trapez).
    """
    height, width = edges_image.shape
    mask = np.zeros_like(edges_image)
    
    # Definim punctele procentual:
    # width * 0.10 = pornește de la 10% din stânga (nu chiar din colț)
    # width * 0.90 = se termină la 90% în dreapta
    # height * 0.35 = linia de sus e la 35% din înălțime (mai sus spre cer)
    
    polygon = np.array([[
        (int(width * 0.10), height),             # Stânga-Jos (puțin strâns)
        (int(width * 0.90), height),             # Dreapta-Jos (puțin strâns)
        (int(width * 0.60), int(height * 0.35)), # Dreapta-Sus (Mai SUS și mai LARG)
        (int(width * 0.40), int(height * 0.35))  # Stânga-Sus (Mai SUS și mai LARG)
    ]], np.int32)

    # Umplem poligonul cu alb (255) pe masca neagră
    cv2.fillPoly(mask, polygon, 255)
    
    # Aplicăm masca peste imaginea cu contururi (Bitwise AND)
    masked_image = cv2.bitwise_and(edges_image, mask)
    
    return masked_image

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preia imaginea color, o face grayscale, reduce zgomotul,
    detectează contururile și aplică masca ROI.
    """
    # 1. Convertește imaginea în grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Aplică un filtru Gaussian pentru a reduce zgomotul
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Aplică detecția de contururi Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # 4. Aplică Region of Interest (ROI) cu noile coordonate
    final_edges = mask_region_of_interest(edges)
    
    return final_edges