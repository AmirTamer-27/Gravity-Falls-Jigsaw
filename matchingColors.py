import cv2
import numpy as np
import glob
import os
from itertools import permutations
import matplotlib.pyplot as plt

# =========================
# 1) Config
# =========================
PIECES_FOLDER = "puzzle_pieces_2x2"
CORRECT_FOLDER = "correct"
ROWS, COLS = 2, 2
BORDER_STRIP_THICKNESS = 1

# =========================
# 2) Helpers
# =========================
def load_specific_pieces(file_paths):
    """Loads a specific list of 4 image paths."""
    pieces = []
    # Sort by filename to ensure piece0, piece1, piece2, piece3 order if possible, 
    # though the solver will reorder them anyway.
    file_paths = sorted(file_paths)
    
    for idx, path in enumerate(file_paths):
        img = cv2.imread(path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pieces.append({
            "id": idx,
            "path": path,
            "rgb": rgb
        })
    return pieces

def extract_borders_color(rgb, k=BORDER_STRIP_THICKNESS):
    H, W, _ = rgb.shape
    k = max(1, min(k, H // 2, W // 2))
    top    = rgb[0:k, :, :]
    bottom = rgb[H-k:H, :, :]
    left   = rgb[:, 0:k, :].transpose(1, 0, 2)
    right  = rgb[:, W-k:W, :].transpose(1, 0, 2)
    return {"top": top, "bottom": bottom, "left": left, "right": right}

def border_diff_color(a, b):
    """L1 distance between two RGB borders"""
    H = min(a.shape[0], b.shape[0])
    W = min(a.shape[1], b.shape[1])
    return float(np.sum(np.abs(a[:H, :W, :].astype(int) - b[:H, :W, :].astype(int))))

def solve_puzzle(pieces):
    """
    Runs the exact logic provided in the original script
    Returns the reconstructed image canvas.
    """
    N = len(pieces)
    if N != 4:
        return None

    # --- Original Logic Start ---
    piece_borders = [extract_borders_color(p["rgb"]) for p in pieces]
    sides = ["top", "bottom", "left", "right"]
    opposite = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}

    # 4) Compute pairwise side-to-side scores
    matches = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            for side_i, side_j in [("right", "left"), ("left", "right"), ("bottom", "top"), ("top", "bottom")]:
                strip_i = piece_borders[i][side_i]
                strip_j = piece_borders[j][side_j]
                score = border_diff_color(strip_i, strip_j)
                matches.append((score, i, side_i, j, side_j))
    matches.sort(key=lambda x: x[0])

    score_dict = {(i, side_i, j, side_j): score for score, i, side_i, j, side_j in matches}
    def side_score(i, side_i, j, side_j, default=1e9):
        return score_dict.get((i, side_i, j, side_j), default)

    # 5) Per-side "outerness"
    best_match_potential = {i:{} for i in range(N)}
    for i in range(N):
        for side in sides:
            best_score = float("inf")
            for j in range(N):
                if i==j: continue
                opp_side = opposite[side]
                score = border_diff_color(piece_borders[i][side], piece_borders[j][opp_side])
                if score < best_score:
                    best_score = score
            best_match_potential[i][side] = best_score

    corner_TL = [best_match_potential[i]["top"] + best_match_potential[i]["left"] for i in range(N)]
    corner_TR = [best_match_potential[i]["top"] + best_match_potential[i]["right"] for i in range(N)]
    corner_BL = [best_match_potential[i]["bottom"] + best_match_potential[i]["left"] for i in range(N)]
    corner_BR = [best_match_potential[i]["bottom"] + best_match_potential[i]["right"] for i in range(N)]

    # 6) Global 2x2 search
    piece_ids = list(range(N))
    best_perm = None
    best_adj_cost = float("inf")
    best_border_score = -float("inf")
    EPS = 1e-6

    for perm in permutations(piece_ids, 4):
        pTL, pTR, pBL, pBR = perm
        adj_cost = (
            side_score(pTL,"right",pTR,"left") + 
            side_score(pBL,"right",pBR,"left") +
            side_score(pTL,"bottom",pBL,"top") +
            side_score(pTR,"bottom",pBR,"top")
        )
        border_score = corner_TL[pTL]+corner_TR[pTR]+corner_BL[pBL]+corner_BR[pBR]
        if adj_cost < best_adj_cost - EPS:
            best_adj_cost = adj_cost
            best_border_score = border_score
            best_perm = perm
        elif abs(adj_cost - best_adj_cost) <= EPS and border_score > best_border_score:
            best_border_score = border_score
            best_perm = perm

    layout = [[best_perm[0], best_perm[1]], [best_perm[2], best_perm[3]]]

    # 7) Reconstruct visualization
    tile_h, tile_w, _ = pieces[0]["rgb"].shape
    canvas = np.zeros((ROWS*tile_h, COLS*tile_w, 3), dtype=np.uint8)

    for r in range(ROWS):
        for c in range(COLS):
            pid = layout[r][c]
            tile = pieces[pid]["rgb"]
            y1, y2 = r*tile_h, (r+1)*tile_h
            x1, x2 = c*tile_w, (c+1)*tile_w
            canvas[y1:y2, x1:x2, :] = tile
    
    return canvas

# =========================
# Main Execution Loop
# =========================

def main():
    # 1. Identify all unique Puzzle IDs from filenames in PIECES_FOLDER
    # Assumes format like: img0_piece0.png
    all_files = glob.glob(os.path.join(PIECES_FOLDER, "*.png"))
    puzzle_ids = set()
    
    for f in all_files:
        basename = os.path.basename(f)
        if "img" in basename and "_piece" in basename:
            # Extract ID between 'img' and '_piece'
            try:
                pid_str = basename.split("img")[1].split("_piece")[0]
                puzzle_ids.add(int(pid_str))
            except ValueError:
                pass
    
    sorted_ids = sorted(list(puzzle_ids))
    print(f"Found {len(sorted_ids)} puzzles to process.")
    
    correct_count = 0
    total_processed = 0

    print(f"{'ID':<5} | {'Status':<10} | {'Error (MSE)'}")
    print("-" * 35)

    for pid in sorted_ids:
        # 2. Gather the 4 pieces for this ID
        piece_pattern = os.path.join(PIECES_FOLDER, f"img{pid}_piece*.png")
        piece_paths = glob.glob(piece_pattern)
        
        if len(piece_paths) != 4:
            print(f"{pid:<5} | SKIP (Missing pieces)")
            continue

        # 3. Load Pieces and Solve
        pieces = load_specific_pieces(piece_paths)
        reconstructed_img = solve_puzzle(pieces)
        
        if reconstructed_img is None:
            print(f"{pid:<5} | ERROR (Solve failed)")
            continue

        # 4. Load Ground Truth
        # Try both .png and .jpg for the correct folder
        gt_path_png = os.path.join(CORRECT_FOLDER, f"{pid}.png")
        gt_path_jpg = os.path.join(CORRECT_FOLDER, f"{pid}.jpg")
        
        ground_truth = None
        if os.path.exists(gt_path_png):
            ground_truth = cv2.imread(gt_path_png)
        elif os.path.exists(gt_path_jpg):
            ground_truth = cv2.imread(gt_path_jpg)
            
        if ground_truth is None:
            print(f"{pid:<5} | NO GT (Ground truth not found)")
            continue
            
        # Convert GT to RGB to match reconstructed format
        ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2RGB)

        # 5. Compare
        # Resize GT if dimensions differ slightly due to cutting rounding errors
        if ground_truth.shape != reconstructed_img.shape:
            ground_truth = cv2.resize(ground_truth, (reconstructed_img.shape[1], reconstructed_img.shape[0]))

        # Calculate Mean Squared Error
        mse = np.mean((reconstructed_img.astype("float") - ground_truth.astype("float")) ** 2)
        
        # We allow a small tolerance for compression artifacts or 1px shifts
        is_correct = mse < 5.0 
        
        status = "✅ PASS" if is_correct else "❌ FAIL"
        if is_correct:
            correct_count += 1
        
        print(f"{pid:<5} | {status:<10} | {mse:.2f}")
        total_processed += 1

    print("-" * 35)
    print(f"Total Correct: {correct_count} / {total_processed}")
    if total_processed > 0:
        print(f"Accuracy: {(correct_count / total_processed) * 100:.2f}%")

if __name__ == "__main__":
    main()