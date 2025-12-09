# inference_test_v3_sdxl.py
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
import torch
from pathlib import Path
from PIL import Image
import argparse
import re

# --- Hardcoded list of prompts ---
PROMPTS = [
    "AINBN3tpDb, moderate Fusarium Head Blight, 55.6%",
    "AINBN3tpDb, severe Fusarium Head Blight, 100.0%",
    "AINBN3tpDb, mild Fusarium Head Blight, 10.0%",
    "AINBN3tpDb, no Fusarium Head Blight, 0.0%",
]

def shorten_prompt(prompt):
    # Remove the first word, then replace spaces with underscores
    words = prompt.split()
    short = "_".join(words[1:])  # Remove first word
    # Remove non-alphanumeric characters except underscores
    short = re.sub(r'[^\w_]', '', short)
    return short[:40]  # Limit length

def main():
    parser = argparse.ArgumentParser(description="Batch inference with Stable Diffusion XL ControlNet")
    parser.add_argument("--num_images", type=int, required=True, help="Number of images to generate")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images")
    parser.add_argument("--conditioning_image", type=str, required=True, help="Path to conditioning image")
    parser.add_argument("--base_model_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="Base SDXL model path")
    parser.add_argument("--controlnet_weights_path", type=str, default="controlnet_training_output/v1_sdxl", help="ControlNet weights path")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load custom ControlNet weights
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_weights_path,
        torch_dtype=torch.float16
    )

    # Load Stable Diffusion XL ControlNet pipeline
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        args.base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")

    conditioning_image = Image.open(args.conditioning_image).convert("RGB")

    for i in range(args.num_images):
        prompt = PROMPTS[i % len(PROMPTS)]
        print(f"Generating image {i+1} with prompt: {prompt}")
        image = pipe(prompt, image=conditioning_image, num_inference_steps=15).images[0]
        short_prompt = shorten_prompt(prompt)
        out_path = output_dir / f"{i+1}_{short_prompt}.png"
        image.save(out_path)
        print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()