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

# Define where our folders are
PIECES_FOLDER = "puzzle_pieces_2x2"
EDGE_FOLDER = "HOGS_2x2"
CORRECT_FOLDER = "correct"

GRID_SIZE = 2

# Settings for the HOG (Histogram of Oriented Gradients) feature
HOG_PIXELS_PER_CELL = (4, 4)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS = 9
HOG_STRIP_DEPTH = 16 

# How much we trust the HOG score vs the Color score
HOG_WEIGHT = 2 

# =========================
# 2) Helper: HOG Extraction
# =========================

def compute_hog_for_edge(image, side):
    """
    1. Grabs a slice (strip) of the image edge.
    2. Runs Sobel Edge Detection to find lines/structures.
    3. Computes the HOG features on those lines.
    """
    
    # Get height and width of the image
    h, w = image.shape
    
    # Decide which strip of the image to take based on the 'side'
    if side == 'top':
        # Take the top rows up to HOG_STRIP_DEPTH
        strip = image[0:HOG_STRIP_DEPTH, :]
    elif side == 'bottom':
        # Take the bottom rows
        strip = image[-HOG_STRIP_DEPTH:, :]
    elif side == 'left':
        # Take the left columns
        strip = image[:, 0:HOG_STRIP_DEPTH]
    elif side == 'right':
        # Take the right columns
        strip = image[:, -HOG_STRIP_DEPTH:]
    else:
        # If something is wrong, return an empty array
        return np.zeros(10) 
        
    # --- Sobel Edge Detection ---
    # This detects vertical lines
    gx = cv2.Sobel(strip, cv2.CV_32F, 1, 0, ksize=3)
    # This detects horizontal lines
    gy = cv2.Sobel(strip, cv2.CV_32F, 0, 1, ksize=3)
    
    # Combine them to get the total edge strength
    edge_map = np.sqrt(gx*gx + gy*gy)
    
    # Normalize the result to be between 0 and 255 (standard image format)
    edge_map = cv2.normalize(edge_map, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert numbers to integers (uint8 is standard for images)
    edge_map = edge_map.astype(np.uint8)

    # Try to compute HOG features. If it fails (e.g., image too small), return zeros.
    try:
        fd = hog(edge_map, 
                 orientations=HOG_ORIENTATIONS, 
                 pixels_per_cell=HOG_PIXELS_PER_CELL,
                 cells_per_block=HOG_CELLS_PER_BLOCK, 
                 block_norm='L2-Hys', 
                 visualize=False)
        return fd
    except:
        return np.zeros(10) # Fallback if math fails

# =========================
# 3) Cost Calculation with Locking + HOG
# =========================

def calculate_costs_with_locking(pieces, edge_pieces):
    """
    Calculates how well every piece fits with every other piece.
    """
    n = len(pieces)
    
    # Create empty matrices to store costs (scores). 
    # n x n matrix filled with zeros initially.
    h_costs = np.zeros((n, n))
    v_costs = np.zeros((n, n))
    
    # 1. Pre-compute LAB colors for all pieces
    # LAB color space matches human vision better than RGB
    lab_pieces = []
    for p in pieces:
        # Convert from BGR (Blue-Green-Red) to LAB
        converted_img = cv2.cvtColor(p["color"], cv2.COLOR_BGR2LAB)
        # Convert to float (decimals) for precise math
        converted_img = converted_img.astype("float32")
        lab_pieces.append(converted_img)
    
    # 2. Pre-compute HOG features for all pieces
    hog_feats = []
    for edge_img in edge_pieces:
        # If the edge image has color (3 channels), turn it into grayscale (1 channel)
        if len(edge_img.shape) == 3:
            edge_img = cv2.cvtColor(edge_img, cv2.COLOR_BGR2GRAY)
            
        # Calculate features for all 4 sides of this piece
        feats = {}
        feats['top'] = compute_hog_for_edge(edge_img, 'top')
        feats['bottom'] = compute_hog_for_edge(edge_img, 'bottom')
        feats['left'] = compute_hog_for_edge(edge_img, 'left')
        feats['right'] = compute_hog_for_edge(edge_img, 'right')
        
        hog_feats.append(feats)
    
    # 3. Calculate costs between every pair of pieces
    for i in range(n):
        for j in range(n):
            # A piece cannot fit with itself
            if i == j: 
                h_costs[i][j] = float('inf')
                v_costs[i][j] = float('inf')
                continue
            
            # --- HORIZONTAL CHECK: Piece i (Left) vs Piece j (Right) ---
            
            # Get the rightmost column of pixels from piece i
            right_col_i = lab_pieces[i][:, -1, :]
            # Get the leftmost column of pixels from piece j
            left_col_j = lab_pieces[j][:, 0, :]
            
            # Calculate Color Difference (average difference)
            color_diff = np.abs(right_col_i - left_col_j)
            color_h = np.sum(color_diff) / lab_pieces[i].shape[0]
            
            # Calculate Shape/Edge Difference using Euclidean distance
            hog_h = euclidean(hog_feats[i]['right'], hog_feats[j]['left'])
            
            # Combine them: Cost = Color + (Shape * Weight)
            h_costs[i][j] = color_h + (hog_h * HOG_WEIGHT)
            
            # --- VERTICAL CHECK: Piece i (Top) vs Piece j (Bottom) ---
            
            # Get the bottom row of pixels from piece i
            bottom_row_i = lab_pieces[i][-1, :, :]
            # Get the top row of pixels from piece j
            top_row_j = lab_pieces[j][0, :, :]
            
            # Calculate Color Difference
            color_diff_v = np.abs(bottom_row_i - top_row_j)
            color_v = np.sum(color_diff_v) / lab_pieces[i].shape[1]
            
            # Calculate Shape/Edge Difference
            hog_v = euclidean(hog_feats[i]['bottom'], hog_feats[j]['top'])
            
            # Combine them
            v_costs[i][j] = color_v + (hog_v * HOG_WEIGHT)

    # --- Locking Logic ---
    # If a match is REALLY good compared to the second best match, "lock" it in.
    locks_made = 0
    
    def apply_desperation(cost_matrix):
        """Looks for very strong matches and lowers their cost significantly."""
        local_locks = 0
        rows, cols = cost_matrix.shape
        
        for i in range(rows):
            # Copy the costs for this row
            row_vals = cost_matrix[i, :].copy()
            
            # Ignore the cost to self (infinity)
            row_vals[i] = float('inf') 
            
            # Find the best match (minimum cost)
            best_j = np.argmin(row_vals)
            best_val = row_vals[best_j]
            
            # Find the second best match
            row_vals[best_j] = float('inf') # Temporarily hide the best
            second_best_val = np.min(row_vals)
            
            # Compare best vs second best
            ratio = best_val / (second_best_val + 1e-9) # +1e-9 avoids division by zero
            
            # If the best is much smaller (better) than the second best
            if ratio < 0.6: 
                # Give it a huge bonus (negative cost)
                cost_matrix[i][best_j] -= 20000.0
                local_locks += 1
                
        return local_locks

    # Apply locking to horizontal and vertical costs
    locks_made += apply_desperation(h_costs)
    locks_made += apply_desperation(v_costs)
    
    print(f"  > Costs (Color+HOG): Locked {locks_made} strong connections.")
    
    return h_costs, v_costs

# =========================
# 4) The Solver (Best-First Search)
# =========================

def solve_puzzle_best_first(pieces, h_costs, v_costs, rows=4, cols=4):
    """
    Tries to solve the puzzle by picking the best fitting pieces one by one.
    """
    N = len(pieces)
    
    # Helper function to try solving starting with a specific piece
    def solve_from_start(start_idx):
        # Create an empty grid
        grid = [[None for _ in range(cols)] for _ in range(rows)]
        
        # Place the first piece at top-left (0,0)
        grid[0][0] = start_idx
        
        # Keep track of pieces we have used
        used = {start_idx}
        pieces_placed = 1
        total_energy = 0
        
        # Keep placing pieces until the grid is full
        while pieces_placed < N:
            best_move = None
            min_global_error = float("inf")
            
            # Look at every spot in the grid
            for r in range(rows):
                for c in range(cols):
                    # If this spot is already filled, skip it
                    if grid[r][c] is not None: 
                        continue 
                    
                    # Find who the neighbors are for this empty spot
                    neighbors = [] 
                    
                    # Is there a neighbor to the left?
                    if c > 0 and grid[r][c-1] is not None: 
                        neighbors.append((grid[r][c-1], "h_forward")) 
                    
                    # Is there a neighbor above?
                    if r > 0 and grid[r-1][c] is not None:
                        neighbors.append((grid[r-1][c], "v_forward"))
                    
                    # Is there a neighbor to the right?
                    if c < cols-1 and grid[r][c+1] is not None:
                        neighbors.append((grid[r][c+1], "h_backward"))
                    
                    # Is there a neighbor below?
                    if r < rows-1 and grid[r+1][c] is not None:
                        neighbors.append((grid[r+1][c], "v_backward"))
                        
                    # If this spot has no neighbors yet, we can't place anything here
                    if len(neighbors) == 0: 
                        continue 
                    
                    # Try every unused piece in this spot
                    for p_idx in range(N):
                        if p_idx in used: 
                            continue
                        
                        current_error = 0
                        
                        # Calculate error based on all existing neighbors
                        for n_idx, relation in neighbors:
                            if relation == "h_forward": 
                                current_error += h_costs[n_idx][p_idx]
                            elif relation == "v_forward": 
                                current_error += v_costs[n_idx][p_idx]
                            elif relation == "h_backward": 
                                current_error += h_costs[p_idx][n_idx]
                            elif relation == "v_backward": 
                                current_error += v_costs[p_idx][n_idx]
                        
                        # If this piece fits better than any other piece we've tried so far
                        if current_error < min_global_error:
                            min_global_error = current_error
                            best_move = (r, c, p_idx)
            
            # If we found a valid move, make it
            if best_move:
                r, c, p_idx = best_move
                grid[r][c] = p_idx
                used.add(p_idx)
                total_energy += min_global_error
                pieces_placed += 1
            else:
                # If we get stuck, stop
                break
                
        return grid, total_energy

    # Try solving the puzzle starting with EVERY possible piece
    best_solution = None
    min_global_energy = float("inf")
    
    for start_node in range(N):
        candidate_grid, energy = solve_from_start(start_node)
        
        # Keep the solution with the lowest total error energy
        if energy < min_global_energy:
            min_global_energy = energy
            best_solution = candidate_grid
            
    # Flatten the best grid into a list
    solution_perm = []
    if best_solution:
        for r in range(rows):
            for c in range(cols):
                solution_perm.append(best_solution[r][c])
            
    return solution_perm

# =========================
# 5) Helpers: Stitch & Find Correct
# =========================

def stitch_image(pieces, layout_indices, rows, cols):
    """
    Takes the solved order of pieces and glues them back into one big image.
    """
    # Get size of one piece
    h, w = pieces[0]["color"].shape[:2]
    
    # Create a blank canvas
    canvas = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
    
    for i, p_idx in enumerate(layout_indices):
        # Calculate row and column from index
        r = i // cols
        c = i % cols
        
        tile = pieces[p_idx]["color"]
        
        # Safety check: resize if the piece is slightly wrong size
        if tile.shape[:2] != (h, w): 
            tile = cv2.resize(tile, (w, h))
            
        # Place the piece onto the canvas
        y_start = r * h
        y_end = (r + 1) * h
        x_start = c * w
        x_end = (c + 1) * w
        
        canvas[y_start:y_end, x_start:x_end] = tile
        
    return canvas

def find_correct_image(folder, pid):
    """
    Tries to find the ground truth (solution) image in the folder.
    It checks a few different filenames just in case.
    """
    if not os.path.exists(folder): 
        return None
        
    patterns = [f"img{pid}_piece0.png", f"{pid}.png", f"{pid}.jpg", f"img{pid}.png"]
    
    for fname in patterns:
        path = os.path.join(folder, fname)
        if os.path.exists(path): 
            return cv2.imread(path)
            
    return None

# =========================
# 6) Main Loop
# =========================

def main():
    # Check if folders exist
    if not os.path.exists(PIECES_FOLDER): 
        return
    if not os.path.exists(EDGE_FOLDER): 
        print("Warning: Edge folder not found")

    # Find all puzzle piece files
    files = glob.glob(os.path.join(PIECES_FOLDER, "*.png"))
    
    # Figure out the Puzzle IDs (e.g. 0, 1, 2...)
    puzzle_ids = set()
    for f in files:
        if "img" in f and "_piece" in f:
            try:
                # Extract the ID number from the filename
                filename = os.path.basename(f)
                parts = filename.split("img")
                # parts[1] is something like "0_piece3.png"
                pid_str = parts[1].split("_piece")[0]
                puzzle_ids.add(int(pid_str))
            except: 
                pass
    
    # Sort the IDs so we process them in order
    sorted_ids = sorted(list(puzzle_ids))
    
    print(f"Processing {len(sorted_ids)} puzzles...")
    print(f"{'ID':<5} | {'Status':<10} | {'MSE Error'}")
    print("-" * 35)

    correct_count = 0
    total_attempted = 0

    for pid in sorted_ids:
        # 1. Load Color Pieces for this specific puzzle ID
        pattern = os.path.join(PIECES_FOLDER, f"img{pid}_piece*.png")
        piece_files = sorted(glob.glob(pattern))
        
        pieces = []
        edge_pieces = []
        
        for fpath in piece_files:
            # Load Color Image
            color_img = cv2.imread(fpath)
            
            # Load Edge Map (for HOG features)
            basename = os.path.basename(fpath)
            edge_path = os.path.join(EDGE_FOLDER, basename)
            
            if os.path.exists(edge_path):
                # Read as grayscale
                edge_img = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
            else:
                # If edge map is missing, make a blank black image as fallback
                h, w = color_img.shape[:2]
                edge_img = np.zeros((h, w), dtype=np.uint8)

            pieces.append({"color": color_img, "path": fpath})
            edge_pieces.append(edge_img)

        # Check if we have enough pieces (2x2 = 4 pieces)
        if len(pieces) < (GRID_SIZE * GRID_SIZE): 
            continue
        
        # --- 2. Calculate Costs ---
        h_costs, v_costs = calculate_costs_with_locking(pieces, edge_pieces)

        # --- 3. Solve ---
        solution_perm = solve_puzzle_best_first(pieces, h_costs, v_costs, rows=GRID_SIZE, cols=GRID_SIZE)
        
        # Rebuild the image
        candidate_img = stitch_image(pieces, solution_perm, rows=GRID_SIZE, cols=GRID_SIZE)
        
        # Find the correct answer to compare
        correct_img = find_correct_image(CORRECT_FOLDER, pid)

        # --- Check Accuracy ---
        status = "⚠️ NO GT" # NO Ground Truth
        mse = -1.0
        
        if correct_img is not None:
            # Ensure sizes match
            if candidate_img.shape != correct_img.shape:
                correct_img = cv2.resize(correct_img, (candidate_img.shape[1], candidate_img.shape[0]))
                
            # Calculate Mean Squared Error (Difference between images)
            diff = candidate_img.astype("float") - correct_img.astype("float")
            mse = np.mean(diff ** 2)
            
            if mse < 150.0:
                status = "✅ PASS"
                correct_count += 1
            else:
                status = "❌ FAIL"
        
        total_attempted += 1
        print(f"{pid:<5} | {status:<10} | {mse:.2f}")

        # --- VISUALIZATION (Optional) ---
        if SHOW_VISUALS:
            # Create a plot with 2 subplots side-by-side
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            # Show Assembled Image
            disp_assembled = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2RGB)
            axes[0].imshow(disp_assembled)
            axes[0].set_title(f"Assembled (ID: {pid})")
            axes[0].axis('off')

            # Show Correct Image
            if correct_img is not None:
                disp_correct = cv2.cvtColor(correct_img, cv2.COLOR_BGR2RGB)
                axes[1].imshow(disp_correct)
                axes[1].set_title(f"Correct ({status})")
            else:
                axes[1].text(0.5, 0.5, "Image Not Found", 
                             horizontalalignment='center', verticalalignment='center', color='red')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.show() 

    print("-" * 35)
    if total_attempted > 0:
        accuracy = (correct_count / total_attempted) * 100
        print(f"Final Accuracy: {correct_count}/{total_attempted} ({accuracy:.2f}%)")

# This means "If you run this file directly, run the main function"
if __name__ == "__main__":
    main()