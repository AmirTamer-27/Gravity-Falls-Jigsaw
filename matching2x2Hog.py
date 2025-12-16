import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from skimage.feature import hog
from scipy.spatial.distance import euclidean

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

# Weights for cost calculation using HOG and Color
HOG_WEIGHT = 1.5
COLOR_WEIGHT = 1



def compute_hog_for_edge(image, side):
    #Gets sent an image and its side so it extracts the strip it needs
    h, w = image.shape
    
    'side'
    if side == 'top':
        strip = image[0:HOG_STRIP_DEPTH, :]
    elif side == 'bottom':
        strip = image[-HOG_STRIP_DEPTH:, :]
    elif side == 'left':
        strip = image[:, 0:HOG_STRIP_DEPTH]
    elif side == 'right':
        strip = image[:, -HOG_STRIP_DEPTH:]
    else:
        return np.zeros(10) 
        
    # Apply Sobel edge detection
    gx = cv2.Sobel(strip, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(strip, cv2.CV_32F, 0, 1, ksize=3)
    
    edge_map = np.sqrt(gx*gx + gy*gy)
    #Scales everything between 0 and 255
    edge_map = cv2.normalize(edge_map, None, 0, 255, cv2.NORM_MINMAX)
    #Converts data into starndard image format
    edge_map = edge_map.astype(np.uint8)

   #Hog function returns an array that represents the histogram arr[0] = 0.5 means that there are values at a certain gradient orientation is that are represented as 0.5 after scaling
    try:
        fd = hog(edge_map, 
                 orientations=HOG_ORIENTATIONS, 
                 pixels_per_cell=HOG_PIXELS_PER_CELL,
                 cells_per_block=HOG_CELLS_PER_BLOCK, 
                 block_norm='L2-Hys', 
                 visualize=False)
        return fd
    except:
        return np.zeros(10)


def calculate_costs_with_locking(pieces, edge_pieces):
      #Calculates costs between all pieces input is the colored pieces and the edge pieces so that both are factored into the cost calculation

   #Create two empty matrices for the costs
    n = len(pieces)
    
    h_costs = np.zeros((n, n))
    v_costs = np.zeros((n, n))
    
   # Convert images to LAB color space Lab color space can match better than normal RGB
    lab_pieces = []
    for p in pieces:
        converted_img = cv2.cvtColor(p["color"], cv2.COLOR_BGR2LAB)
        converted_img = converted_img.astype("float32")
        lab_pieces.append(converted_img)
    
    # Compute HOG features for all edges
    hog_feats = []
    for edge_img in edge_pieces:
        feats = {}
        feats['top'] = compute_hog_for_edge(edge_img, 'top')
        feats['bottom'] = compute_hog_for_edge(edge_img, 'bottom')
        feats['left'] = compute_hog_for_edge(edge_img, 'left')
        feats['right'] = compute_hog_for_edge(edge_img, 'right')
        
        hog_feats.append(feats)
    
    # Calculate costs
    for i in range(n):
        for j in range(n):
             #Skip if we're at the same image
            if i == j: 
                h_costs[i][j] = float('inf')
                v_costs[i][j] = float('inf')
                continue
            
            # Check the cost between right of piece i and left of piece j , in the color space and also the HOGs
            right_col_i = lab_pieces[i][:, -1, :]
            left_col_j = lab_pieces[j][:, 0, :]
            color_diff = np.abs(right_col_i - left_col_j)
            color_h = np.sum(color_diff) / lab_pieces[i].shape[0]
            hog_h = euclidean(hog_feats[i]['right'], hog_feats[j]['left'])
            h_costs[i][j] = color_h * COLOR_WEIGHT + (hog_h * HOG_WEIGHT)
            
            # Check the cost between bottom of piece i and top of piece j , in the color space and also the HOGs
            bottom_row_i = lab_pieces[i][-1, :, :]
            top_row_j = lab_pieces[j][0, :, :]
            color_diff_v = np.abs(bottom_row_i - top_row_j)
            color_v = np.sum(color_diff_v) / lab_pieces[i].shape[1]
            hog_v = euclidean(hog_feats[i]['bottom'], hog_feats[j]['top'])
            v_costs[i][j] = color_v * COLOR_WEIGHT + (hog_v * HOG_WEIGHT)

    def apply_desperation(cost_matrix):
        local_locks = 0
        rows, cols = cost_matrix.shape
        
        for i in range(rows):
            # Copy the costs for this row
            row_vals = cost_matrix[i, :].copy()
            row_vals[i] = float('inf') 
            
            # Find the best match
            best_j = np.argmin(row_vals)
            best_val = row_vals[best_j]
            
            # Find the second best match
            row_vals[best_j] = float('inf') 
            second_best_val = np.min(row_vals)
            
            ratio = best_val / (second_best_val + 1e-9) 
            
            # If the best is much smaller than the second best
            if ratio < 0.6: 
                # Decrease its cost majorly to be the best choice
                cost_matrix[i][best_j] -= 20000.0
                local_locks += 1
                
        return local_locks

    # Apply locking to horizontal and vertical costs
    apply_desperation(h_costs)
    apply_desperation(v_costs)
    
    return h_costs, v_costs


def solve_puzzle_best_first(pieces, h_costs, v_costs, rows=4, cols=4):
    """
    Tries to solve the puzzle by picking the best fitting pieces one by one.
    """
    N = len(pieces)
    
 
    def solve_from_start(start_idx):
       # Initialize grid
        grid = [[None for _ in range(cols)] for _ in range(rows)]
        grid[0][0] = start_idx
        used = {start_idx}
        pieces_placed = 1
        total_score = 0
        
      
        while pieces_placed < N:
            best_move = None
            min_global_error = float("inf")
            
            # Look for all empty places where u can possibly place new piece , check were this new piece will be places right to the current , left , top or bottom then add to candidate slots
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] is not None: 
                        continue 
                    neighbors = [] 
                    if c > 0 and grid[r][c-1] is not None: 
                        neighbors.append((grid[r][c-1], "left")) 
                    if r > 0 and grid[r-1][c] is not None:
                        neighbors.append((grid[r-1][c], "top"))
                    if c < cols-1 and grid[r][c+1] is not None:
                        neighbors.append((grid[r][c+1], "right"))
                    if r < rows-1 and grid[r+1][c] is not None:
                        neighbors.append((grid[r+1][c], "bottom"))
                        
                    # append the row and column or th cell we will place at and all the neighbors to it.
                    if len(neighbors) == 0: 
                        continue 
                    
                    # Try every unused piece in this spot
                    for p_idx in range(N):
                        if p_idx in used: 
                            continue
                        
                        current_error = 0
                        
                        # Calculate error based on all existing neighbors
                        for n_idx, relation in neighbors:
                            if relation == "left": 
                                current_error += h_costs[n_idx][p_idx]
                            elif relation == "top": 
                                current_error += v_costs[n_idx][p_idx]
                            elif relation == "right": 
                                current_error += h_costs[p_idx][n_idx]
                            elif relation == "bottom": 
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
                total_score += min_global_error
                pieces_placed += 1
            else:
                # If we get stuck, stop
                break
                
        return grid, total_score

    # Try solving the puzzle starting with EVERY possible piece
    best_solution = None
    min_score = float("inf")
    
    for start_node in range(N):
        candidate_grid, score = solve_from_start(start_node)
        
        # Keep the solution with the lowest total error score
        if score < min_score:
            min_score = score
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


def main():
    if not os.path.exists(PIECES_FOLDER): 
        return
    if not os.path.exists(EDGE_FOLDER): 
        print("Warning: Edge folder not found")

    files = glob.glob(os.path.join(PIECES_FOLDER, "*.png"))
    puzzle_ids = set()
    for f in files:
        if "img" in f and "_piece" in f:
            try:
                filename = os.path.basename(f)
                parts = filename.split("img")
                pid_str = parts[1].split("_piece")[0]
                puzzle_ids.add(int(pid_str))
            except: 
                pass
    
    sorted_ids = sorted(list(puzzle_ids))
    
    print(f"Processing {len(sorted_ids)} puzzles...")
    print(f"{'ID':<5} | {'Status':<10} | {'MSE Error'}")
    print("-" * 35)

    correct_count = 0
    total_attempted = 0

    for pid in sorted_ids:
        pattern = os.path.join(PIECES_FOLDER, f"img{pid}_piece*.png")
        piece_files = sorted(glob.glob(pattern))
        
        pieces = []
        edge_pieces = []
        
        for fpath in piece_files:
            color_img = cv2.imread(fpath)
            
            basename = os.path.basename(fpath)
            edge_path = os.path.join(EDGE_FOLDER, basename)
            
            if os.path.exists(edge_path):
                edge_img = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
            else:
                h, w = color_img.shape[:2]
                edge_img = np.zeros((h, w), dtype=np.uint8)

            pieces.append({"color": color_img, "path": fpath})
            edge_pieces.append(edge_img)

        if len(pieces) < (GRID_SIZE * GRID_SIZE): 
            continue
        
        h_costs, v_costs = calculate_costs_with_locking(pieces, edge_pieces)

        solution_perm = solve_puzzle_best_first(pieces, h_costs, v_costs, rows=GRID_SIZE, cols=GRID_SIZE)
        
        candidate_img = stitch_image(pieces, solution_perm, rows=GRID_SIZE, cols=GRID_SIZE)
        
        correct_img = find_correct_image(CORRECT_FOLDER, pid)

        status = "⚠️ NO GT" 
        mse = -1.0
        
        if correct_img is not None:
            if candidate_img.shape != correct_img.shape:
                correct_img = cv2.resize(correct_img, (candidate_img.shape[1], candidate_img.shape[0]))
                
            diff = candidate_img.astype("float") - correct_img.astype("float")
            mse = np.mean(diff ** 2)
            
            if mse < 150.0:
                status = "✅ PASS"
                correct_count += 1
            else:
                status = "❌ FAIL"
        
        total_attempted += 1
        print(f"{pid:<5} | {status:<10} | {mse:.2f}")

        if SHOW_VISUALS:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            disp_assembled = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2RGB)
            axes[0].imshow(disp_assembled)
            axes[0].set_title(f"Assembled (ID: {pid})")
            axes[0].axis('off')

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

if __name__ == "__main__":
    main()