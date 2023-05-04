from __future__ import annotations

import random
import tempfile
import imageio
import numpy as np
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler


from utils import generate_video_name_mp4


def to_video(frames: list[np.ndarray], fps: int, out_file=None) -> str:
    if out_file is None:
        out_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    writer = imageio.get_writer(out_file, format="FFMPEG", fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    return out_file


class ModelscopeT2V:
    def __init__(self, device):
        pipe = DiffusionPipeline.from_pretrained(
            "damo-vilab/text-to-video-ms-1.7b",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(device)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()

        self.pipe = pipe

    def generate_video(
        self,
        prompt: str,
        seed: int,
        num_frames: int,
        num_inference_steps: int,
        out_file: str = None,
    ) -> str:
        if seed == -1:
            seed = random.randint(0, 1000000)
        generator = torch.Generator().manual_seed(seed)
        frames = self.pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            generator=generator,
        ).frames
        return to_video(frames, 8, out_file=out_file)

    def inference(self, inputs):
        video_path = generate_video_name_mp4()
        self.generate_video(
            prompt=inputs,
            seed=-1,
            num_frames=16,
            num_inference_steps=25,
            out_file=video_path,
        )
        return video_path
