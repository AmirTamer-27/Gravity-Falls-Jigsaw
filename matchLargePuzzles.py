import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from skimage.feature import hog
from scipy.spatial.distance import euclidean

# =========================
# 1) Configuration
# =========================

SHOW_VISUALS = False

PIECES_FOLDER = "puzzle_pieces_4x4"
CORRECT_FOLDER = "correct"


GRID_SIZE = 4

# =========================
# 2) Cost Calculation with Locking
# =========================

def calculate_costs_with_locking(pieces):
    """
    Calculates pairwise costs but applies STRICT LOCKING for 'Best Buddies'.
    Returns Horizontal and Vertical cost matrices.
    """
    n = len(pieces)
    h_costs = np.zeros((n, n))
    v_costs = np.zeros((n, n))
    
    # Pre-convert to LAB for speed
    lab_pieces = [cv2.cvtColor(p["color"], cv2.COLOR_BGR2LAB).astype("float32") for p in pieces]
    
    # --- 1. Standard LAB Calculation ---
    for i in range(n):
        for j in range(n):
            if i == j: 
                h_costs[i][j] = float('inf')
                v_costs[i][j] = float('inf')
                continue
            
            # Horizontal Cost: Right col of i vs Left col of j
            # Normalized by size (height * 1 channel)
            diff_h = np.sum(np.abs(lab_pieces[i][:, -1, :] - lab_pieces[j][:, 0, :])) / lab_pieces[i].shape[0]
            h_costs[i][j] = diff_h
            
            # Vertical Cost: Bottom row of i vs Top row of j
            # Normalized by size (width * 1 channel)
            diff_v = np.sum(np.abs(lab_pieces[i][-1, :, :] - lab_pieces[j][0, :, :])) / lab_pieces[i].shape[1]
            v_costs[i][j] = diff_v

    # Helper for locking logic
    locks_made = 0
    
    def apply_desperation(cost_matrix):
        local_locks = 0
        rows, cols = cost_matrix.shape
        
        # Check every piece 'i'
        for i in range(rows):
            row_vals = cost_matrix[i, :].copy()
            row_vals[i] = float('inf') # Ignore self
            
            # Find best match
            best_j = np.argmin(row_vals)
            best_val = row_vals[best_j]
            
            # Find second best
            row_vals[best_j] = float('inf')
            second_best_val = np.min(row_vals)
            
            # Ratio Test
            # If Best is < 40% of the cost of Second Best, it's a "Desperate" match.
            # (Strict threshold to avoid false positives)
            ratio = best_val / (second_best_val + 1e-9)
            
            if ratio < 0.6: 
                # FORCE LOCK: Subtract massive bonus
                # We don't check if 'j' likes 'i' back. 'i' has no other choice.
                cost_matrix[i][best_j] -= 20000.0
                local_locks += 1
                
        return local_locks

    locks_made += apply_desperation(h_costs)
    locks_made += apply_desperation(v_costs)
                
    print(f"  > Optimized Costs: Locked {locks_made} strong connections.")
    return h_costs, v_costs

# =========================
# 3) The Solver (Best-First / Prim's Approach)
# =========================

def solve_puzzle_best_first(pieces, h_costs, v_costs, rows=4, cols=4):
    """
    Solves the puzzle using 'Best-First' strategy using pre-calculated costs.
    """
    N = len(pieces)
    if N != rows * cols:
        print(f"Error: Expected {rows*cols} pieces, found {N}")
        return None

    # --- Inner Solver: Best-First Growth ---
    def solve_from_start(start_idx):
        grid = [[None for _ in range(cols)] for _ in range(rows)]
        grid[0][0] = start_idx
        used = {start_idx}
        
        pieces_placed = 1
        total_energy = 0
        
        while pieces_placed < N:
            best_move = None
            min_global_error = float("inf")
            
            # Scan the entire grid for valid empty spots (Frontier)
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] is not None: continue # Already filled
                    
                    # Check neighbors
                    neighbors = [] 
                    
                    # Look at existing neighbors to calculate cost
                    # Note on lookup: h_costs[A][B] means A is Left, B is Right.
                    
                    if c > 0 and grid[r][c-1] is not None: 
                        # Neighbor is LEFT. We want: h_costs[Neighbor][Candidate]
                        neighbors.append((grid[r][c-1], "h_forward")) 
                        
                    if r > 0 and grid[r-1][c] is not None:
                        # Neighbor is TOP. We want: v_costs[Neighbor][Candidate]
                        neighbors.append((grid[r-1][c], "v_forward"))
                        
                    if c < cols-1 and grid[r][c+1] is not None:
                        # Neighbor is RIGHT. We want: h_costs[Candidate][Neighbor]
                        neighbors.append((grid[r][c+1], "h_backward"))
                        
                    if r < rows-1 and grid[r+1][c] is not None:
                        # Neighbor is BOTTOM. We want: v_costs[Candidate][Neighbor]
                        neighbors.append((grid[r+1][c], "v_backward"))
                        
                    if len(neighbors) == 0: continue 
                    
                    # Try all unused pieces in this specific spot
                    for p_idx in range(N):
                        if p_idx in used: continue
                        
                        current_error = 0
                        
                        # Sum up error with ALL existing neighbors
                        for n_idx, relation in neighbors:
                            if relation == "h_forward":
                                current_error += h_costs[n_idx][p_idx]
                            elif relation == "v_forward":
                                current_error += v_costs[n_idx][p_idx]
                            elif relation == "h_backward":
                                current_error += h_costs[p_idx][n_idx]
                            elif relation == "v_backward":
                                current_error += v_costs[p_idx][n_idx]
                        
                        if current_error < min_global_error:
                            min_global_error = current_error
                            best_move = (r, c, p_idx)
            
            # Execute the best move found across the ENTIRE board
            if best_move:
                r, c, p_idx = best_move
                grid[r][c] = p_idx
                used.add(p_idx)
                total_energy += min_global_error
                pieces_placed += 1
            else:
                break
                            
        return grid, total_energy

    # --- Main Loop: Try Every Piece as Top-Left ---
    best_solution = None
    min_global_energy = float("inf")
    
    for start_node in range(N):
        candidate_grid, energy = solve_from_start(start_node)
        
        if energy < min_global_energy:
            min_global_energy = energy
            best_solution = candidate_grid
            
    # Flatten
    solution_perm = []
    if best_solution:
        for r in range(rows):
            for c in range(cols):
                solution_perm.append(best_solution[r][c])
            
    return solution_perm

# =========================
# 4) Reconstruction Helper
# =========================

def stitch_image(pieces, layout_indices, rows, cols):
    h, w = pieces[0]["color"].shape[:2]
    canvas = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
    for i, p_idx in enumerate(layout_indices):
        r = i // cols
        c = i % cols
        tile = pieces[p_idx]["color"]
        th, tw = tile.shape[:2]
        if th != h or tw != w:
            tile = cv2.resize(tile, (w, h))
        canvas[r*h:(r+1)*h, c*w:(c+1)*w] = tile
    return canvas

# =========================
# 5) Helper: Find Correct Image
# =========================

def find_correct_image(folder, pid):
    if not os.path.exists(folder):
        return None

    patterns = [
        f"img{pid}_piece0.png",
        f"{pid}.png",       
        f"{pid}.jpg",       
        f"img{pid}.png",    
        f"image{pid}.png"   
    ]
    
    for fname in patterns:
        full_path = os.path.join(folder, fname)
        if os.path.exists(full_path):
            return cv2.imread(full_path)
    
    return None

# =========================
# 6) Main Loop
# =========================

def main():
    print(f"\n--- Checking Correct Folder: {CORRECT_FOLDER} ---")
    if os.path.exists(CORRECT_FOLDER):
        files_inside = os.listdir(CORRECT_FOLDER)
        print(f"✅ Folder found! It contains {len(files_inside)} files.")
    else:
        print(f"❌ ERROR: Folder not found at {os.path.abspath(CORRECT_FOLDER)}")
        return
    print("-" * 40 + "\n")

    if not os.path.exists(PIECES_FOLDER):
        print(f"Error: Folder '{PIECES_FOLDER}' not found.")
        return

    files = glob.glob(os.path.join(PIECES_FOLDER, "*.png"))
    puzzle_ids = set()
    for f in files:
        basename = os.path.basename(f)
        if "img" in basename and "_piece" in basename:
            try:
                pid = basename.split("img")[1].split("_piece")[0]
                if pid.isdigit(): puzzle_ids.add(int(pid))
            except: pass
    
    sorted_ids = sorted(list(puzzle_ids))
    print(f"Found {len(sorted_ids)} puzzles (Running Best-First + Locking).\n")
    print(f"{'ID':<5} | {'Status':<10} | {'MSE Error'}")
    print("-" * 35)

    correct_count = 0
    total_attempted = 0

    for pid in sorted_ids:
        # Load pieces
        piece_files = sorted(glob.glob(os.path.join(PIECES_FOLDER, f"img{pid}_piece*.png")))
        
        pieces = []
        for fpath in piece_files:
            img = cv2.imread(fpath)
            if img is not None:
                pieces.append({"color": img, "path": fpath})

        if len(pieces) < (GRID_SIZE * GRID_SIZE):
            continue
        
        # --- 1. CALCULATE COSTS WITH LOCKING ---
        h_costs, v_costs = calculate_costs_with_locking(pieces)

        # --- 2. SOLVE USING BEST-FIRST METHOD ---
        solution_perm = solve_puzzle_best_first(pieces, h_costs, v_costs, rows=GRID_SIZE, cols=GRID_SIZE)
        
        candidate_img = stitch_image(pieces, solution_perm, rows=GRID_SIZE, cols=GRID_SIZE)

        # Find Correct Image
        correct_img = find_correct_image(CORRECT_FOLDER, pid)

        # --- ACCURACY CALCULATION ---
        status = "⚠️ NO GT"
        error_val = -1.0
        
        if correct_img is not None:
            if candidate_img.shape != correct_img.shape:
                correct_img = cv2.resize(correct_img, (candidate_img.shape[1], candidate_img.shape[0]))
            
            mse = np.mean((candidate_img.astype("float") - correct_img.astype("float")) ** 2)
            error_val = mse
            
            if mse < 20.0:
                status = "✅ PASS"
                correct_count += 1
            else:
                status = "❌ FAIL"
        
        total_attempted += 1
        print(f"{pid:<5} | {status:<10} | {error_val:.2f}")

        # --- VISUALIZATION ---
        if SHOW_VISUALS:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            disp_assembled = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2RGB)
            axes[0].imshow(disp_assembled)
            axes[0].set_title(f"Assembled (ID: {pid})")
            axes[0].axis('off')

            if correct_img is not None:
                disp_correct = cv2.cvtColor(correct_img, cv2.COLOR_BGR2RGB)
                axes[1].imshow(disp_correct)
                axes[1].set_title(f"Correct ({status}) MSE: {error_val:.1f}")
            else:
                axes[1].text(0.5, 0.5, "Image Not Found", 
                             horizontalalignment='center', verticalalignment='center', color='red')
            axes[1].axis('off')
            plt.tight_layout()
            plt.show() 

    # --- FINAL SUMMARY ---
    print("-" * 35)
    if total_attempted > 0:
        acc = (correct_count / total_attempted) * 100
        print(f"Final Accuracy: {correct_count}/{total_attempted} ({acc:.2f}%)")
    else:
        print("No puzzles were attempted.")

if __name__ == "__main__":
    main()