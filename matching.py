import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from itertools import permutations
import math

# =========================
# 1) Config
# =========================

EDGE_FOLDER = "edge_maps"
ROWS, COLS = 2, 2
BORDER_WEIGHT = 2.0     # how strongly we prefer bad-matching outer borders
DILATION_ITER = 1       # dilation iterations before comparing borders
BORDER_STRIP_THICKNESS = 5


# =========================
# 2) Helpers
# =========================

def load_gray_pieces(folder):
    exts = ["*.png", "*.jpg", "*.jpeg"]
    paths = sorted([p for ext in exts for p in glob.glob(os.path.join(folder, ext))])

    pieces = []
    for idx, path in enumerate(paths):
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pieces.append({
            "id": idx,
            "path": path,
            "gray": gray
        })
    return pieces


def extract_borders(gray, k=BORDER_STRIP_THICKNESS):
    """
    Return border strips: top, bottom, left, right
    left/right are transposed to have shape (k, length)
    """
    H, W = gray.shape
    k = max(1, min(k, H // 2, W // 2))

    top    = gray[0:k, :]
    bottom = gray[H-k:H, :]

    left   = gray[:, 0:k].T
    right  = gray[:, W-k:W].T

    return {"top": top, "bottom": bottom, "left": left, "right": right}


def border_diff(a, b):
    """
    Difference between two border strips.
    Uses dilation + L1 distance. Lower = better match.
    """
    kernel = np.ones((3, 3), np.uint8)

    a = cv2.dilate(a, kernel, iterations=DILATION_ITER).astype(np.float32)
    b = cv2.dilate(b, kernel, iterations=DILATION_ITER).astype(np.float32)

    # crop to common size just in case
    H = min(a.shape[0], b.shape[0])
    W = min(a.shape[1], b.shape[1])
    a = a[:H, :W]
    b = b[:H, :W]

    return float(np.sum(np.abs(a - b)))


# =========================
# 3) Load pieces & borders
# =========================

pieces = load_gray_pieces(EDGE_FOLDER)
N = len(pieces)

if N != 4:
    print(f"⚠️ Found {N} pieces, but this script assumes a 2x2 puzzle with 4 pieces.")

piece_borders = []
for p in pieces:
    piece_borders.append(extract_borders(p["gray"]))

sides = ["top", "bottom", "left", "right"]
opposite = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}

# =========================
# 4) Compute pairwise side-to-side scores
# =========================

# list of (score, i, side_i, j, side_j)
matches = []

for i in range(N):
    for j in range(N):
        if i == j:
            continue
        for side_i, side_j in [
            ("right", "left"),
            ("left", "right"),
            ("bottom", "top"),
            ("top", "bottom")
        ]:
            strip_i = piece_borders[i][side_i]
            strip_j = piece_borders[j][side_j]
            score = border_diff(strip_i, strip_j)
            matches.append((score, i, side_i, j, side_j))

# sort matches so best (smallest) are first
matches.sort(key=lambda x: x[0])

print(f"Computed {len(matches)} side-to-side scores.")
print("\nTop 10 best side matches:")
for k in range(min(10, len(matches))):
    score, i, si, j, sj = matches[k]
    print(f"{k+1:2d}. Piece {i} {si:6s} ↔ Piece {j} {sj:6s} | score={score:.2f}")


# Build lookup: (i, side_i, j, side_j) -> score
score_dict = {}
for score, i, side_i, j, side_j in matches:
    score_dict[(i, side_i, j, side_j)] = score

def side_score(i, side_i, j, side_j, default=1e9):
    return score_dict.get((i, side_i, j, side_j), default)


# =========================
# 5) Per-side "outerness" (how bad it matches anyone)
# =========================

best_match_potential = {i: {} for i in range(N)}

for i in range(N):
    for side in sides:
        best_score = float("inf")
        for j in range(N):
            if i == j:
                continue
            opp_side = opposite[side]
            score = border_diff(piece_borders[i][side], piece_borders[j][opp_side])
            if score < best_score:
                best_score = score
        best_match_potential[i][side] = best_score

print("\nPer-piece, per-side best match scores (small = good internal edge, big = good outer edge):")
for i in range(N):
    bt = best_match_potential[i]["top"]
    bb = best_match_potential[i]["bottom"]
    bl = best_match_potential[i]["left"]
    br = best_match_potential[i]["right"]
    print(f"Piece {i}: top={bt:.1f}, bottom={bb:.1f}, left={bl:.1f}, right={br:.1f}")


# For each piece and each corner position, how "outer" does it look there?
# (larger = better for being on the border in that corner)
corner_TL = [best_match_potential[i]["top"]    + best_match_potential[i]["left"]   for i in range(N)]
corner_TR = [best_match_potential[i]["top"]    + best_match_potential[i]["right"]  for i in range(N)]
corner_BL = [best_match_potential[i]["bottom"] + best_match_potential[i]["left"]   for i in range(N)]
corner_BR = [best_match_potential[i]["bottom"] + best_match_potential[i]["right"]  for i in range(N)]

print("\nCorner outerness scores (bigger = more likely that corner):")
for i in range(N):
    print(
        f"Piece {i}: "
        f"TL={corner_TL[i]:.1f}, TR={corner_TR[i]:.1f}, "
        f"BL={corner_BL[i]:.1f}, BR={corner_BR[i]:.1f}"
    )


# =========================
# 6) Global 2x2 search with border-aware scoring
# =========================

piece_ids = list(range(N))
best_perm = None
best_adj_cost = float("inf")
best_border_score = -float("inf")
EPS = 1e-6

# perm = (pTL, pTR, pBL, pBR)
for perm in permutations(piece_ids, 4):
    pTL, pTR, pBL, pBR = perm

    # --- adjacency (internal seams) ---
    adj_cost = 0.0
    # horizontal seams
    adj_cost += side_score(pTL, "right",  pTR, "left")
    adj_cost += side_score(pBL, "right",  pBR, "left")
    # vertical seams
    adj_cost += side_score(pTL, "bottom", pBL, "top")
    adj_cost += side_score(pTR, "bottom", pBR, "top")

    # --- border score: how "outer" each side of the puzzle looks ---
    border_score = (
        corner_TL[pTL] +
        corner_TR[pTR] +
        corner_BL[pBL] +
        corner_BR[pBR]
    )

    # total cost: minimize adjacency, break ties by maximizing border_score
    if adj_cost < best_adj_cost - EPS:
        best_adj_cost = adj_cost
        best_border_score = border_score
        best_perm = perm
    elif abs(adj_cost - best_adj_cost) <= EPS and border_score > best_border_score:
        best_border_score = border_score
        best_perm = perm

print("\nBest permutation (row-major 2x2):", best_perm)
print("Best adjacency cost :", best_adj_cost)
print("Best border score   :", best_border_score)

layout = [
    [best_perm[0], best_perm[1]],  # [top-left, top-right]
    [best_perm[2], best_perm[3]]   # [bottom-left, bottom-right]
]

print("\nFinal layout (indices):")
for r in range(ROWS):
    print(layout[r])


# =========================
# 7) Reconstruct visualization
# =========================

tile_h, tile_w = pieces[0]["gray"].shape
canvas = np.zeros((ROWS * tile_h, COLS * tile_w), dtype=np.uint8)

for r in range(ROWS):
    for c in range(COLS):
        pid = layout[r][c]
        tile = pieces[pid]["gray"]
        y1 = r * tile_h
        y2 = (r + 1) * tile_h
        x1 = c * tile_w
        x2 = (c + 1) * tile_w
        canvas[y1:y2, x1:x2] = tile

plt.figure(figsize=(4, 4))
plt.imshow(canvas, cmap="gray")
plt.title("Reconstructed puzzle (edge maps, border-aware + dilation)")
plt.axis("off")
plt.show()
