import torch
import os
from .shape_gen import ShapE
from tqdm import tqdm

class Shap_E:
    def __init__(self, device):
        self.cond_type = "t2m"
        self.cache_dir = "modules/shap_e/cache_dir"
        self.output_dir = "video"
        self.device = device
        
    def inference(self, prompt):
        self.cond = prompt
        shapE = ShapE(device=self.device, cache_dir=self.cache_dir, type=self.cond_type)

        if os.path.exists(self.cond):
            if self.cond_type == "t2m":
                prompts_path = self.cond
                with open(prompts_path, "r") as f:
                    prompts = f.readlines()
                prompts = [prompt.strip() for prompt in prompts]

                for prompt in tqdm(prompts):
                    results_dir = shapE.inference(prompt, self.output_dir)

            elif self.cond_type == "i2m":
                base_dir = self.cond
                images_path = [
                    os.path.join(base_dir, f)
                    for f in os.listdir(base_dir)
                    if f.endswith(".png") or f.endswith(".jpg")
                ]

                for image_path in tqdm(images_path):
                    results_dir = shapE.inference(image_path, self.output_dir)

        else:
            results_dir = shapE.inference(self.cond, self.output_dir)
            print(f"Output saved to {results_dir}")
        
    