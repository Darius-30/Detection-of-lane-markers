import numpy as np
from scipy.ndimage import maximum_filter

def find_peaks(accumulator: np.ndarray, thetas: np.ndarray, threshold_rel: float = 0.4, neighborhood_size: int = 20):
    """
    Găsește vârfurile locale, dar ignoră liniile orizontale.
    """
    # 1. Prag absolut bazat pe maximul global
    global_max = np.max(accumulator)
    if global_max == 0:
        return []
    
    abs_threshold = global_max * threshold_rel
    
    # 2. Găsește maximele locale
    local_max = maximum_filter(accumulator, size=neighborhood_size) == accumulator
    
    # 3. Mască de intensitate (elimină zgomotul slab)
    detected_peaks_mask = local_max & (accumulator > abs_threshold)
    
    # --- FILTRARE UNGHIURI ---
    # Eliminăm liniile care sunt prea aproape de orizontală (aprox 90 grade).
    # Păstrăm doar liniile verticale/diagonale.
    # Theta 0 sau 180 = Vertical | Theta 90 = Orizontal
    
    # Convertim thetas din radiani în grade pentru a fi mai ușor de înțeles
    thetas_deg = np.rad2deg(thetas)
    
    # Creăm o mască pentru unghiuri "bune"
    # Păstrăm: [0...70] (Stânga) ȘI [110...180] (Dreapta)
    # Eliminăm: [70...110] (Zona Orizontală)
    angle_mask = (thetas_deg < 70) | (thetas_deg > 110)
    
    # Replicăm masca de unghiuri pe toate rândurile (rho) pentru a se potrivi cu acumulatorul
    # angle_mask este 1D (theta), accumulator este 2D (rho, theta)
    full_angle_mask = np.tile(angle_mask, (accumulator.shape[0], 1))
    
    # Combinăm masca de vârfuri cu masca de unghiuri
    final_mask = detected_peaks_mask & full_angle_mask
    
    # FINAL 
    
    # Extragem coordonatele
    peaks_indices = np.argwhere(final_mask)
    
    return [tuple(p) for p in peaks_indices]