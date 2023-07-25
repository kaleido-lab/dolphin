import os
import shutil
import torch
import argparse
import imageio

from tqdm import tqdm

from utils import get_new_uuid

from .shap_e.diffusion.sample import sample_latents
from .shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from .shap_e.models.download import load_model, load_config
from .shap_e.util.notebooks import (
    create_pan_cameras,
    decode_latent_images,
    decode_latent_mesh,
)
from .shap_e.util.image_util import load_image


class ShapE:
    def __init__(self, device, cache_dir, type="i2m"):
        model_name = "image300M" if type == "i2m" else "text300M"

        self.type = type
        self.device = device
        self.xm = load_model("transmitter", device=device, cache_dir=cache_dir)
        self.model = load_model(model_name, device=device, cache_dir=cache_dir)
        self.diffusion = diffusion_from_config(
            load_config("diffusion", cache_dir=cache_dir)
        )

    def inference(
        self, condition, output_dir, batch_size=1, guidance_scale=None, render=True
    ):  # guidance_scale = 15.0 is recommended for text2mesh
        if self.type == "i2m":
            # To get the best result, you should remove the background and show only the object of interest to the model.
            image_path = condition
            image = load_image(image_path)
            model_kwargs = dict(images=[image] * batch_size)
            item_name = os.path.basename(image_path).split(".")[0]
            # copy images into output dir
            shutil.copy(image_path, output_dir)

            guidance_scale = 3.0
        elif self.type == "t2m":
            prompt = condition
            model_kwargs = dict(texts=[prompt] * batch_size)
            item_name = prompt.replace(" ", "_")

            guidance_scale = 12.0

        latents = sample_latents(
            batch_size=batch_size,
            model=self.model,
            diffusion=self.diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=model_kwargs,
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )

        for i, latent in enumerate(latents):
            t = decode_latent_mesh(self.xm, latent).tri_mesh()

            with open(f"{output_dir}/{item_name}---{i}.obj", "w") as f:
                t.write_obj(f)

        if render:
            render_mode = "nerf"  # you can change this to 'stf' for mesh rendering
            size = 256  # this is the size of the renders; higher values take longer to render.

            cameras = create_pan_cameras(size, self.device)
            for i, latent in enumerate(latents):
                images = decode_latent_images(
                    self.xm, latent, cameras, rendering_mode=render_mode
                )
                uuid = get_new_uuid()
                gif_filename = "image/" + uuid + ".gif"
                
                print("filename is:", gif_filename)
                images[0].save(
                    gif_filename,
                    save_all=True,
                    append_images=images[1:],
                    duration=100,
                    loop=0,
                )
                
                # change gif to mp4
                gif_images = imageio.mimread(gif_filename)
                
                video_filename = "video/" + uuid + ".mp4"
                with imageio.get_writer(video_filename, format='FFMPEG', mode='I', fps=10) as writer:
                    for frame in gif_images:
                        writer.append_data(frame)

        return output_dir


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--cond_type",
#         type=str,
#         default="t2m",
#         choices=["i2m", "t2m"],
#         help="i2m(image2mesh) or t2m(text2mesh), default is i2m",
#     )
#     parser.add_argument(
#         "--cond",
#         type=str,
#         required=True,
#         help="can be a prompt string or a file containing prompts split with newlines for t2m, \
#         or path of an image or a directory containing images for i2m",
#     )
#     parser.add_argument(
#         "--cache_dir",
#         type=str,
#         default="modeling",
#         help="model cache dir for shap-e",
#     )
#     parser.add_argument("--output_dir", type=str, default="modeling")
#     args = parser.parse_args()

#     os.makedirs(args.output_dir, exist_ok=True)

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     shapE = ShapE(device=device, cache_dir=args.cache_dir, type=args.cond_type)

#     if os.path.exists(args.cond):
#         if args.cond_type == "t2m":
#             prompts_path = args.cond
#             with open(prompts_path, "r") as f:
#                 prompts = f.readlines()
#             prompts = [prompt.strip() for prompt in prompts]

#             for prompt in tqdm(prompts):
#                 results_dir = shapE.inference(prompt, args.output_dir)

#         elif args.cond_type == "i2m":
#             base_dir = args.cond
#             images_path = [
#                 os.path.join(base_dir, f)
#                 for f in os.listdir(base_dir)
#                 if f.endswith(".png") or f.endswith(".jpg")
#             ]

#             for image_path in tqdm(images_path):
#                 results_dir = shapE.inference(image_path, args.output_dir)

#     else:
#         results_dir = shapE.inference(args.cond, args.output_dir)
#         print(f"Output saved to {results_dir}")

#     print("Done!")
