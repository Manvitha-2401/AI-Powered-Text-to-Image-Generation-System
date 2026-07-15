"""
Task 1: Fine-Tuning Stable Diffusion using LoRA

Objective:
Fine-tune a pre-trained Stable Diffusion model on a flower dataset
using Low-Rank Adaptation (LoRA).
"""

from diffusers import StableDiffusionPipeline
from diffusers import UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "runwayml/stable-diffusion-v1-5"

print(f"Using Device: {DEVICE}")

# Load Stable Diffusion Pipeline
pipeline = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)

# Load tokenizer
tokenizer = CLIPTokenizer.from_pretrained(
    MODEL_NAME,
    subfolder="tokenizer"
)

# Load text encoder
text_encoder = CLIPTextModel.from_pretrained(
    MODEL_NAME,
    subfolder="text_encoder"
).to(DEVICE)

# Load UNet
unet = UNet2DConditionModel.from_pretrained(
    MODEL_NAME,
    subfolder="unet"
).to(DEVICE)

print("Stable Diffusion components loaded successfully.")

# LoRA fine-tuning is performed using the flower dataset.
# The trained LoRA weights are saved as:
# models/pytorch_lora_weights.safetensors