import numpy as np

def solve_line_endpoints(peak_rho_idx, peak_theta_idx, accumulator, thetas, rhos, delta_deg=3):
    # ... (partea de inițializare rămâne la fel) ...
    rho_res = rhos[1] - rhos[0]
    theta_res = thetas[1] - thetas[0]
    delta_idx = int(np.deg2rad(delta_deg) / theta_res)
    num_rhos, num_thetas = accumulator.shape
    theta_l_idx = peak_theta_idx - delta_idx
    theta_r_idx = peak_theta_idx + delta_idx
    
    if theta_l_idx < 0 or theta_r_idx >= num_thetas: return None

    # --- Funcție de Scanare STRICTĂ ---
    def get_rho_bounds_strict(theta_i, center_rho_approx):
        col = accumulator[:, theta_i]
        search_radius = 60
        start_search = max(0, center_rho_approx - search_radius)
        end_search = min(num_rhos, center_rho_approx + search_radius)
        
        local_slice = col[start_search:end_search]
        if np.max(local_slice) == 0: return None, None
        
        local_max_offset = np.argmax(local_slice)
        actual_wing_center = start_search + local_max_offset
        max_val = col[actual_wing_center]
        
        # !!! SCHIMBARE CRITICĂ 1: Prag de 70% !!!
        # Linia se oprește imediat ce intensitatea scade puțin.
        threshold = max_val * 0.70  
        
        # Scanare sus
        curr = actual_wing_center
        while curr > 0 and col[curr] > threshold: curr -= 1
        rho_min_idx = curr
        
        # Scanare jos
        curr = actual_wing_center
        while curr < num_rhos - 1 and col[curr] > threshold: curr += 1
        rho_max_idx = curr
        
        if (rho_max_idx - rho_min_idx) < 2: return None, None
        return rho_min_idx, rho_max_idx

    # ... (apelurile get_rho_bounds rămân la fel) ...
    rho_l_min_idx, rho_l_max_idx = get_rho_bounds_strict(theta_l_idx, peak_rho_idx)
    rho_r_min_idx, rho_r_max_idx = get_rho_bounds_strict(theta_r_idx, peak_rho_idx)
    
    if None in [rho_l_min_idx, rho_l_max_idx, rho_r_min_idx, rho_r_max_idx]: return None

    # ... (conversia și matricele A/B rămân la fel) ...
    theta_l = thetas[theta_l_idx]
    theta_r = thetas[theta_r_idx]
    rho_l_min = rhos[rho_l_min_idx]
    rho_l_max = rhos[rho_l_max_idx]
    rho_r_min = rhos[rho_r_min_idx]
    rho_r_max = rhos[rho_r_max_idx]
    
    A_matrix = np.array([[np.cos(theta_l), np.sin(theta_l)], [np.cos(theta_r), np.sin(theta_r)]])
    B_vector_1 = np.array([rho_l_min, rho_r_max])
    B_vector_2 = np.array([rho_l_max, rho_r_min])
    
    try:
        pt1 = np.linalg.solve(A_matrix, B_vector_1)
        pt2 = np.linalg.solve(A_matrix, B_vector_2)
        
        p1_int = (int(pt1[0]), int(pt1[1]))
        p2_int = (int(pt2[0]), int(pt2[1]))

        # !!! SCHIMBARE CRITICĂ 2: Filtrul "Anti-Cer" !!!
        # Dacă un punct este în jumătatea de sus a imaginii (unde e cerul/munții),
        # înseamnă că linia a fost proiectată greșit. O ștergem.
        # Presupunem că imaginea are aprox 720 înălțime. Orice y < 250 e suspect.
        if p1_int[1] < 250 or p2_int[1] < 250:
            return None
            
        return p1_int, p2_int
        
    except np.linalg.LinAlgError:
        return None