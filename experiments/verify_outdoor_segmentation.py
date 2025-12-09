import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os

# python verify_outdoor_segmentation.py --image_path wheat_inference_data/frame_0011.jpg --detections_path outputs/colour_transfer_testing/segmentations/frame_0011_detections.npz
# python verify_outdoor_segmentation.py --image_path wheat_inference_data/green_wheat_02.jpeg --detections_path outputs/colour_transfer_testing/segmentations/green_wheat_02_detections.npz
# python verify_outdoor_segmentation.py --image_path wheat_inference_data/green_wheat_08.jpg --detections_path outputs/colour_transfer_testing/segmentations/green_wheat_08_detections.npz
# python verify_outdoor_segmentation.py --image_path wheat_inference_data/green_wheat_11.jpg --detections_path outputs/colour_transfer_testing/segmentations/green_wheat_11_detections.npz
# python verify_outdoor_segmentation.py --image_path wheat_inference_data/frame_0011.jpg --detections_path outputs/colour_transfer_testing/segmentations/frame_0011_detections.npz --output_path outputs/verification/frame_0011_result.png
def main():
    parser = argparse.ArgumentParser(description="Inspect and visualize wheat head detections and masks.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the original outdoor image")
    parser.add_argument("--detections_path", type=str, required=True, help="Path to the .npz detections file")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the visualization (e.g., outputs/verification/result.png)")
    parser.add_argument("--no_display", action="store_true", help="Don't display plots interactively, only save")
    args = parser.parse_args()

    # Load image and detections
    image = np.array(Image.open(args.image_path).convert("RGB"))
    data = np.load(args.detections_path)
    masks = data["masks"]
    bboxes = data["bboxes"]
    confidences = data["confidences"]

    print(f"Number of masks: {masks.shape[0]}")
    print(f"Mask shape: {masks.shape[1:]}")
    print(f"Bounding boxes:\n{bboxes}")
    print(f"Confidences:\n{confidences}")

    # === VISUALIZE ALL MASKS AT ONCE ===
    fig = plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    # Create a combined mask with different colors for each wheat head
    combined_mask = np.zeros((*masks.shape[1:], 3), dtype=np.float32)
    
    # Generate distinct colors for each mask
    colors = plt.cm.tab20(np.linspace(0, 1, len(masks)))
    
    for i, mask in enumerate(masks):
        # Create colored mask
        mask_bool = mask > 0
        for c in range(3):
            combined_mask[:, :, c] += mask_bool * colors[i, c]
    
    # Normalize combined mask
    max_val = combined_mask.max()
    if max_val > 0:
        combined_mask = combined_mask / max_val
    
    plt.imshow(combined_mask, alpha=0.5)
    plt.title(f"All {len(masks)} Wheat Heads Detected")
    plt.axis("off")
    plt.tight_layout()
    
    # Save if output path is provided
    if args.output_path:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        plt.savefig(args.output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {args.output_path}")
    
    # Display if not suppressed
    if not args.no_display:
        plt.show()
    else:
        plt.close(fig)

    # === VISUALIZE EACH MASK INDIVIDUALLY ===
    if not args.no_display:
        for i, mask in enumerate(masks):
            plt.figure()
            plt.imshow(image)
            plt.imshow(mask, alpha=0.5, cmap="jet")
            plt.title(f"Wheat Head {i+1} (Conf: {confidences[i]:.2f})")
            plt.axis("off")
            plt.show()

if __name__ == "__main__":
    main()