import os
import glob
from PIL import Image

# ================= CONFIGURATION =================
# Folder containing your 110 original images (e.g., "raw_images" or ".")
INPUT_FOLDER = "puzzle_4x4_F" 

# Choose your puzzle size here: 2, 4, or 8
# 2  -> 2x2 grid (4 pieces)
# 4  -> 4x4 grid (16 pieces)
# 8  -> 8x8 grid (64 pieces)
GRID_SIZE = 4
# =================================================

def process_batch(input_folder, grid_n):
    # 1. Setup Output Folder
    output_folder = f"puzzle_pieces_{grid_n}x{grid_n}"
    os.makedirs(output_folder, exist_ok=True)
    
    # 2. Get all images (jpg, png, jpeg)
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
    
    # Sort files to keep processing order clean (0.jpg, 1.jpg...)
    # We try to sort numerically if filenames are numbers
    try:
        image_files.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
    except:
        image_files.sort()

    print(f"Found {len(image_files)} images in '{input_folder}'. Processing for {grid_n}x{grid_n} grid...")

    count_processed = 0

    for file_path in image_files:
        try:
            img = Image.open(file_path)
            w, h = img.size
            
            # Extract ID from filename (e.g. "0.jpg" -> "0")
            basename = os.path.basename(file_path)
            img_id = os.path.splitext(basename)[0]

            # Calculate piece size
            piece_w = w // grid_n
            piece_h = h // grid_n

            if piece_w == 0 or piece_h == 0:
                print(f"Skipping {basename}: Image too small for {grid_n}x{grid_n} split.")
                continue

            # Crop Loop
            piece_idx = 0
            for row in range(grid_n):
                for col in range(grid_n):
                    left = col * piece_w
                    upper = row * piece_h
                    right = left + piece_w
                    lower = upper + piece_h
                    
                    # Crop
                    piece = img.crop((left, upper, right, lower))
                    
                    # Save Name: imgX_pieceY.png
                    save_name = f"img{img_id}_piece{piece_idx}.png"
                    save_path = os.path.join(output_folder, save_name)
                    
                    piece.save(save_path, format="PNG")
                    piece_idx += 1
            
            count_processed += 1
            # print(f"Processed {basename} -> {piece_idx} pieces") # Uncomment for verbose

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"\n✅ Done! Processed {count_processed} images.")
    print(f"Pieces saved in folder: {output_folder}/")

# Run the function
if __name__ == "__main__":
    # Ensure your images are in the folder defined in INPUT_FOLDER
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Error: Input folder '{INPUT_FOLDER}' does not exist.")
        print("Please create a folder named 'raw_images' and put your 110 jpg files inside.")
    else:
        process_batch(INPUT_FOLDER, GRID_SIZE)