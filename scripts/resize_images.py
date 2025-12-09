import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm  # For progress bar

def resize_image_with_padding(input_path, output_path, size=(512, 512), fill_color=(128, 128, 128)):
    # Read the image
    img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Warning: Could not read {input_path}")
        return False

    # Normalize color format
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] != 3:
        print(f"Warning: {input_path} has unexpected channels ({img.shape[2]}). Skipping.")
        return False

    h, w = img.shape[:2]
    target_h, target_w = size

    # Calculate aspect ratio and new dimensions
    w_ratio = target_w / w
    h_ratio = target_h / h
    scale = min(w_ratio, h_ratio)

    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize with LANCZOS4 interpolation
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Create the final square canvas and center the resized image
    canvas = np.full((target_h, target_w, 3), fill_color, dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    # Paste the resized image onto the canvas
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    # Save the final image
    cv2.imwrite(str(output_path), canvas)

    # Verify output dimensions
    saved_img = cv2.imread(str(output_path))
    if saved_img is None or saved_img.shape[:2] != size:
        print(f"Error: Output image {output_path} has incorrect dimensions or failed to save.")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Resize images with aspect ratio preservation (Scale-to-Fit).")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save resized images")
    parser.add_argument("--fill_color", type=int, nargs=3, default=(128, 128, 128), help="RGB fill color for padding (default: 128,128,128)")
    parser.add_argument("--width", type=int, required=True, help="Target width for resized images")
    parser.add_argument("--height", type=int, required=True, help="Target height for resized images")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    size = (args.height, args.width)

    # Get list of valid image files
    image_files = [p for p in input_dir.glob("*") if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
    
    # Process images with progress bar
    for img_path in tqdm(image_files, desc="Resizing images"):
        output_path = output_dir / img_path.name
        success = resize_image_with_padding(img_path, output_path, size=(args.height, args.width), fill_color=args.fill_color)
        if not success:
            print(f"Failed to process {img_path}")

if __name__ == "__main__":
    main()