import numpy as np

def solve_line_endpoints(peak_rho_idx, peak_theta_idx, accumulator, thetas, rhos, delta_deg=4):
    """
    Reconstrucție optimizată pentru a preveni liniile 'prea lungi'.
    Folosește o scanare continuă din centru pentru a ignora zgomotul distant.
    """
    rho_res = rhos[1] - rhos[0]
    theta_res = thetas[1] - thetas[0]
    
    delta_idx = int(np.deg2rad(delta_deg) / theta_res)
    num_rhos, num_thetas = accumulator.shape
    
    theta_l_idx = peak_theta_idx - delta_idx
    theta_r_idx = peak_theta_idx + delta_idx
    
    if theta_l_idx < 0 or theta_r_idx >= num_thetas:
        return None

    # --- Funcție Nouă: Scanare Centrifugă ---
    def get_rho_bounds_strict(theta_i, center_rho_approx):
        col = accumulator[:, theta_i]
        
        # 1. Găsim 'Centrul Local' al aripii
        # Deoarece aripa e înclinată, centrul ei la theta_i nu e fix la peak_rho_idx.
        # Căutăm cel mai puternic punct într-o zonă rezonabilă (+/- 50 pixeli) jurul vârfului original.
        search_radius = 50
        start_search = max(0, center_rho_approx - search_radius)
        end_search = min(num_rhos, center_rho_approx + search_radius)
        
        if start_search >= end_search: return None, None
        
        # Decupăm zona de interes
        local_slice = col[start_search:end_search]
        if np.max(local_slice) == 0: return None, None
        
        # Găsim maximul local real în această felie
        local_max_offset = np.argmax(local_slice)
        actual_wing_center = start_search + local_max_offset
        max_val = col[actual_wing_center]
        
        # 2. Setăm un prag strict (50% din intensitatea aripii)
        # Dacă punem 0.2 (20%), linia iese lungă. Dacă punem 0.6 (60%), iese scurtă.
        threshold = max_val * 0.40 
        
        # 3. Scanăm ÎN SUS (spre indici mai mici) până dăm de o valoare mică
        curr = actual_wing_center
        while curr > 0 and col[curr] > threshold:
            curr -= 1
        rho_min_idx = curr
        
        # 4. Scanăm ÎN JOS (spre indici mai mari) până dăm de o valoare mică
        curr = actual_wing_center
        while curr < num_rhos - 1 and col[curr] > threshold:
            curr += 1
        rho_max_idx = curr
        
        # Verificare de siguranță: Dacă aripa e prea subțire (1-2 pixeli), e zgomot
        if (rho_max_idx - rho_min_idx) < 2:
            return None, None

        return rho_min_idx, rho_max_idx

    # Aplicăm scanarea strictă
    rho_l_min_idx, rho_l_max_idx = get_rho_bounds_strict(theta_l_idx, peak_rho_idx)
    rho_r_min_idx, rho_r_max_idx = get_rho_bounds_strict(theta_r_idx, peak_rho_idx)
    
    if None in [rho_l_min_idx, rho_l_max_idx, rho_r_min_idx, rho_r_max_idx]:
        return None

    # --- Conversie și Rezolvare (Identic cu varianta anterioară) ---
    theta_l = thetas[theta_l_idx]
    theta_r = thetas[theta_r_idx]
    
    rho_l_min = rhos[rho_l_min_idx]
    rho_l_max = rhos[rho_l_max_idx]
    rho_r_min = rhos[rho_r_min_idx]
    rho_r_max = rhos[rho_r_max_idx]
    
    A_matrix = np.array([
        [np.cos(theta_l), np.sin(theta_l)],
        [np.cos(theta_r), np.sin(theta_r)]
    ])
    
    # Împerechere încrucișată (Cross-over)
    B_vector_1 = np.array([rho_l_min, rho_r_max])
    B_vector_2 = np.array([rho_l_max, rho_r_min])
    
    try:
        pt1 = np.linalg.solve(A_matrix, B_vector_1)
        pt2 = np.linalg.solve(A_matrix, B_vector_2)
        
        # Verificare extra: Dacă punctele sunt absurd de departe (ex: 20000 pixeli), le ignorăm
        if abs(pt1[0]) > 5000 or abs(pt1[1]) > 5000: return None

        return (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1]))
        
    except np.linalg.LinAlgError:
        return None