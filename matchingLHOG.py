import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from skimage.feature import hog
from scipy.spatial.distance import euclidean

SHOW_VISUALS = False


PIECES_FOLDER = "puzzle_pieces_8x8"
EDGE_FOLDER = "HOGS_8x8"
CORRECT_FOLDER = "correct"

GRID_SIZE = 8

# HOG Parameters
HOG_PIXELS_PER_CELL = (4, 4)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS = 9
HOG_STRIP_DEPTH = 16 

# Weights for cost calculation using HOG and Color
HOG_WEIGHT = 5
COLOR_WEIGHT = 1


def compute_hog_for_edge(image, side):
   #Gets sent an image and its side so it extracts the strip it needs
    h, w = image.shape
    
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


def calculateCosts(pieces, edge_pieces):
   #Calculates costs between all pieces input is the colored pieces and the edge pieces so that both are factored into the cost calculation

   #Create two empty matrices for the costs
    n = len(pieces)
    h_costs = np.zeros((n, n))
    v_costs = np.zeros((n, n))
    
    # Convert images to LAB color space Lab color space can match better than normal RGB
    lab_pieces = []
    for p in pieces:
        lab_img = cv2.cvtColor(p["color"], cv2.COLOR_BGR2LAB).astype("float32")
        lab_pieces.append(lab_img)
    
    # Compute HOG features for all edges
    hog_feats = []
    for edge_img in edge_pieces:
        feats = {
            'top': compute_hog_for_edge(edge_img, 'top'),
            'bottom': compute_hog_for_edge(edge_img, 'bottom'),
            'left': compute_hog_for_edge(edge_img, 'left'),
            'right': compute_hog_for_edge(edge_img, 'right')
        }
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
            color_h = np.sum(np.abs(lab_pieces[i][:, -1, :] - lab_pieces[j][:, 0, :])) / lab_pieces[i].shape[0]
            hog_h = euclidean(hog_feats[i]['right'], hog_feats[j]['left'])
            h_costs[i][j] = color_h *COLOR_WEIGHT+ (hog_h * HOG_WEIGHT)
            
            # Check the cost between bottom of piece i and top of piece j , in the color space and also the HOGs
            color_v = np.sum(np.abs(lab_pieces[i][-1, :, :] - lab_pieces[j][0, :, :])) / lab_pieces[i].shape[1]
            hog_v = euclidean(hog_feats[i]['bottom'], hog_feats[j]['top'])
            v_costs[i][j] = color_v * COLOR_WEIGHT + (hog_v * HOG_WEIGHT)
    return h_costs, v_costs


def solve_puzzle_best_first(pieces, h_costs, v_costs, rows=4, cols=4):
   
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
            min_global_ratio = float("inf")
            
            # Look for all empty places where u can possibly place new piece , check were this new piece will be places right to the current , left , top or bottom then add to candidate slots
            candidate_slots = []
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] is not None: continue 
                    
                    neighbors = [] 
                    if c > 0 and grid[r][c-1] is not None: 
                        neighbors.append((grid[r][c-1], "left")) 
                    if r > 0 and grid[r-1][c] is not None:
                        neighbors.append((grid[r-1][c], "top"))
                    if c < cols-1 and grid[r][c+1] is not None:
                        neighbors.append((grid[r][c+1], "right"))
                    if r < rows-1 and grid[r+1][c] is not None:
                        neighbors.append((grid[r+1][c], "bottom"))
                        
                    if len(neighbors) > 0:
                        # append the row and column or th cell we will place at and all the neighbors to it.
                        candidate_slots.append((r, c, neighbors))

            if not candidate_slots: break

            # Evaluate matches for each open slot
            for r, c, neighbors in candidate_slots:
                candidates_for_slot = []
                #Itterate over all unused pieces and find their scores with the neighboring piece
                for p_idx in range(N):
                    if p_idx in used: continue
                    
                    current_error = 0
                    valid_connection = True
                    #Check the cost of putting this piece in the empty slot by checking the costs of all the neighbors to that slot
                    for n_idx, relation in neighbors:
                        cost = float('inf')
                        if relation == "left": cost = h_costs[n_idx][p_idx]
                        elif relation == "top": cost = v_costs[n_idx][p_idx]
                        elif relation == "right": cost = h_costs[p_idx][n_idx]
                        elif relation == "bottom": cost = v_costs[p_idx][n_idx]
                        
                        if cost == float('inf'): 
                            valid_connection = False
                            break
                        current_error += cost
                    
                    if valid_connection:
                        candidates_for_slot.append((current_error, p_idx))

                # If no candidates fit, skip this slot
                if not candidates_for_slot: continue
                
                # Sort the candidates by the lowest error
                candidates_for_slot.sort(key=lambda x: x[0])
                
                best_error, best_pid = candidates_for_slot[0]
                
                # Calculate uniqueness ratio
                if len(candidates_for_slot) > 1:
                    second_best_error = candidates_for_slot[1][0]
                    if second_best_error <= 1e-5: second_best_error = 1e-5 
                    ratio = best_error / second_best_error
                else:
                    ratio = -999.0 # Only one option? It's infinitely unique.

                # Keep the move with the best ratio found so far
                if ratio < min_global_ratio:
                    min_global_ratio = ratio
                    best_move = (r, c, best_pid, best_error)

            # Apply the best move found
            if best_move:
                r, c, p_idx, error = best_move
                grid[r][c] = p_idx
                used.add(p_idx)
                total_score += error
                pieces_placed += 1
            else:
                break
                
        return grid, total_score

    # Here we itterate over all the possible starts and build their puzzles the use best solution variable to update based on the error
    best_solution = None
    min_score = float("inf")

    for start_node in range(N):
        #Compare min_score with the score we just code
        candidate_grid, score = solve_from_start(start_node)
        if score < min_score:
            min_score = score
            best_solution = candidate_grid
            
    # Flatten grid to list
    solution_perm = []
    if best_solution:
        for r in range(rows):
            for c in range(cols):
                solution_perm.append(best_solution[r][c])
            
    return solution_perm


def stitch_image(pieces, layout_indices, rows, cols):
    h, w = pieces[0]["color"].shape[:2]
    canvas = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
    for i, p_idx in enumerate(layout_indices):
        r = i // cols
        c = i % cols
        tile = pieces[p_idx]["color"]
        if tile.shape[:2] != (h, w): tile = cv2.resize(tile, (w, h))
        canvas[r*h:(r+1)*h, c*w:(c+1)*w] = tile
    return canvas

def find_correct_image(folder, pid):
    if not os.path.exists(folder): return None
    patterns = [f"img{pid}_piece0.png", f"{pid}.png", f"{pid}.jpg", f"img{pid}.png"]
    for fname in patterns:
        path = os.path.join(folder, fname)
        if os.path.exists(path): return cv2.imread(path)
    return None

def main():
    if not os.path.exists(PIECES_FOLDER): return
    if not os.path.exists(EDGE_FOLDER): print("Warning: Edge folder not found")

    files = glob.glob(os.path.join(PIECES_FOLDER, "*.png"))
    puzzle_ids = set()
    for f in files:
        if "img" in f and "_piece" in f:
            try:
                pid = os.path.basename(f).split("img")[1].split("_piece")[0]
                puzzle_ids.add(int(pid))
            except: pass
    
    sorted_ids = sorted(list(puzzle_ids))
    print(f"Processing {len(sorted_ids)} puzzles...")
    print(f"{'ID':<5} | {'Status':<10} | {'MSE Error'}")
    print("-" * 35)

    correct_count = 0
    total_attempted = 0

    for pid in sorted_ids:
        # Load Pieces
        piece_files = sorted(glob.glob(os.path.join(PIECES_FOLDER, f"img{pid}_piece*.png")))
        
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

        if len(pieces) < (GRID_SIZE * GRID_SIZE): continue
        
        # Calculate Costs
        h_costs, v_costs = calculateCosts(pieces, edge_pieces)

        # Solve
        solution_perm = solve_puzzle_best_first(pieces, h_costs, v_costs, rows=GRID_SIZE, cols=GRID_SIZE)
        
        # Verify
        candidate_img = stitch_image(pieces, solution_perm, rows=GRID_SIZE, cols=GRID_SIZE)
        correct_img = find_correct_image(CORRECT_FOLDER, pid)

        status = "⚠️ NO GT"
        mse = -1.0
        if correct_img is not None:
            if candidate_img.shape != correct_img.shape:
                correct_img = cv2.resize(correct_img, (candidate_img.shape[1], candidate_img.shape[0]))
            mse = np.mean((candidate_img.astype("float") - correct_img.astype("float")) ** 2)
            if mse < 90.0:
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
        acc = (correct_count / total_attempted) * 100
        print(f"Final Accuracy: {correct_count}/{total_attempted} ({acc:.2f}%)")

if __name__ == "__main__":
    main()