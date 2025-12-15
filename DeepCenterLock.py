import cv2
import numpy as np
import glob
import os
import itertools
import sys

# Suppress libpng warnings
os.environ["CV_LOG_LEVEL"] = "0"

# ================= CONFIGURATION =================
PIECES_FOLDER = "puzzle_pieces_2x2"
RESULTS_FOLDER = "solved_puzzles_v10"
GROUND_TRUTH_FOLDER = "correct"
# =================================================

def get_rotations_lab(img):
    rotations_bgr = []
    rotations_lab = []
    current_bgr = img
    current_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype("float32")
    
    for _ in range(4):
        rotations_bgr.append(current_bgr)
        rotations_lab.append(current_lab)
        current_bgr = cv2.rotate(current_bgr, cv2.ROTATE_90_CLOCKWISE)
        current_lab = cv2.rotate(current_lab, cv2.ROTATE_90_CLOCKWISE)
        
    return {"bgr": rotations_bgr, "lab": rotations_lab}

def calculate_seam_cost(img1, img2, direction):
    # Standard L1 Match (Robust)
    if direction == 'h':
        border1 = img1[:, -1, :]
        border2 = img2[:, 0, :]
    elif direction == 'v':
        border1 = img1[-1, :, :]
        border2 = img2[0, :, :]
    return np.mean(np.abs(border1 - border2))

def get_edge_stats(img, side):
    """
    Returns (Variance, Mean_Luminance) for a specific edge.
    """
    if side == "top":    pixels = img[0, :, 0] # L channel only
    elif side == "bottom": pixels = img[-1, :, 0]
    elif side == "left":   pixels = img[:, 0, 0]
    elif side == "right":  pixels = img[:, -1, 0]
    
    return np.std(pixels), np.mean(pixels)

def check_deep_center_consistency(pTL, pTR, pBL, pBR):
    """
    3x3 Deep Center Check
    """
    R = 3
    patch_tl = pTL[-R:, -R:, :]
    patch_tr = pTR[-R:, :R, :]
    patch_bl = pBL[:R, -R:, :]
    patch_br = pBR[:R, :R, :]
    
    # Horizontal & Vertical continuity
    err_h = np.mean(np.abs(patch_tl[:, -1, :] - patch_tr[:, 0, :])) + \
            np.mean(np.abs(patch_bl[:, -1, :] - patch_br[:, 0, :]))
    err_v = np.mean(np.abs(patch_tl[-1, :, :] - patch_bl[0, :, :])) + \
            np.mean(np.abs(patch_tr[-1, :, :] - patch_br[0, :, :]))
            
    # Diagonal Cross-Check
    err_diag = np.mean(np.abs(patch_tl[-1, -1, :] - patch_br[0, 0, :])) + \
               np.mean(np.abs(patch_tr[-1, 0, :] - patch_bl[0, -1, :]))
    
    return err_h + err_v + (err_diag * 0.5)

def solve_puzzle_surgical(pieces_data):
    best_score = float('inf')
    best_assembly = None
    
    indices = range(4)
    
    for perm in itertools.permutations(indices):
        for rots in itertools.product(range(4), repeat=4):
            
            pTL = pieces_data[perm[0]]["lab"][rots[0]]
            pTR = pieces_data[perm[1]]["lab"][rots[1]]
            pBL = pieces_data[perm[2]]["lab"][rots[2]]
            pBR = pieces_data[perm[3]]["lab"][rots[3]]
            
            # 1. Base Match Score
            match_score = 0
            match_score += calculate_seam_cost(pTL, pTR, 'h')
            match_score += calculate_seam_cost(pBL, pBR, 'h')
            match_score += calculate_seam_cost(pTL, pBL, 'v')
            match_score += calculate_seam_cost(pTR, pBR, 'v')
            
            if match_score > best_score: continue
            
            # 2. SURGICAL PENALTY: "The Letterbox Killer"
            penalty = 0
            internal_edges = [
                (pTL, "right"), (pTR, "left"),
                (pBL, "right"), (pBR, "left"),
                (pTL, "bottom"), (pBL, "top"),
                (pTR, "bottom"), (pBR, "top")
            ]
            
            for p, side in internal_edges:
                var, mean_lum = get_edge_stats(p, side)
                if var < 5.0 and mean_lum < 30.0:
                    penalty += 5000.0 
                    
            center_variance = 0
            for p, side in internal_edges:
                var, _ = get_edge_stats(p, side)
                center_variance += var
            
            # 4. Center Lock (Geometry)
            center_lock = check_deep_center_consistency(pTL, pTR, pBL, pBR)
            
            final_score = match_score + penalty + (center_lock * 5.0) - (center_variance * 1.5)
            
            if final_score < best_score:
                best_score = final_score
                
                # Reconstruct
                rTL = pieces_data[perm[0]]["bgr"][rots[0]]
                rTR = pieces_data[perm[1]]["bgr"][rots[1]]
                rBL = pieces_data[perm[2]]["bgr"][rots[2]]
                rBR = pieces_data[perm[3]]["bgr"][rots[3]]
                
                top = np.hstack([rTL, rTR])
                bot = np.hstack([rBL, rBR])
                best_assembly = np.vstack([top, bot])
                
    return best_assembly

def draw_seam_overlays(image):
    """
    Draws GREEN lines over the internal seams for the report requirement.
    """
    vis = image.copy()
    h, w, _ = vis.shape
    cv2.line(vis, (w//2, 0), (w//2, h), (0, 255, 0), 2)
    cv2.line(vis, (0, h//2), (w, h//2), (0, 255, 0), 2)
    return vis

def main():
    if not os.path.exists(PIECES_FOLDER):
        print(f"Error: {PIECES_FOLDER} not found.")
        return
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    
    # Create a separate folder for the "Visualizations" with lines
    VIS_FOLDER = os.path.join(RESULTS_FOLDER, "visualizations")
    os.makedirs(VIS_FOLDER, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(PIECES_FOLDER, "*.png")))
    puzzles = {}
    for f in files:
        try:
            pid = os.path.basename(f).split("img")[1].split("_piece")[0]
            if pid not in puzzles: puzzles[pid] = []
            puzzles[pid].append(cv2.imread(f))
        except: pass
        
    print(f"Running Solver v10 (Final Submission Mode) on {len(puzzles)} puzzles...")
    
    correct_count = 0
    total = 0
    
    for pid, raw_pieces in puzzles.items():
        if len(raw_pieces) != 4: continue
        total += 1
        
        pieces_data = [get_rotations_lab(p) for p in raw_pieces]
        result_img = solve_puzzle_surgical(pieces_data)
        
        # --- VERIFICATION ---
        status = "FAIL"
        mse = -1.0
        gt_path = os.path.join(GROUND_TRUTH_FOLDER, f"{pid}.png")
        if not os.path.exists(gt_path): gt_path = os.path.join(GROUND_TRUTH_FOLDER, f"{pid}.jpg")
        
        if os.path.exists(gt_path):
            gt = cv2.imread(gt_path)
            if gt.shape != result_img.shape:
                gt = cv2.resize(gt, (result_img.shape[1], result_img.shape[0]))
            
            mse = np.mean((result_img.astype("float") - gt.astype("float")) ** 2)
            if mse < 5.0:
                status = "PASS"
                correct_count += 1
                
        print(f"Puzzle {pid}: {status} (MSE: {mse:.2f})")
        
        # # 1. Save Clean Result (SAVES EVERYTHING NOW)
        # cv2.imwrite(os.path.join(RESULTS_FOLDER, f"{pid}_{status}.png"), result_img)
        
        # # 2. Save Visualized Result (SAVES EVERYTHING NOW)
        # vis_img = draw_seam_overlays(result_img)
        # cv2.imwrite(os.path.join(VIS_FOLDER, f"{pid}_visualized.png"), vis_img)

    print("-" * 30)
    acc = (correct_count/total)*100
    print(f"FINAL ACCURACY: {correct_count}/{total} ({acc:.1f}%)")
    print("-" * 30)

if __name__ == "__main__":
    main()