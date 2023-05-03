import torch
import numpy as np
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from PIL import Image

from video_utils import prepare_video


class ImageCaptioning:
    def __init__(self, device):
        print("Initializing BLIP2 for ImageCaptioning")
        self.device = device
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        ).to(device)

    def image_captioning(self, image: Image, prompt=None, is_vqa=False):
        if prompt and is_vqa:
            prompt = f"Question: {prompt} Answer:"
        inputs = self.processor(image, text=prompt, return_tensors="pt").to(
            self.device, torch.float16
        )
        generated_ids = self.model.generate(**inputs, max_new_tokens=40)
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        return generated_text

    def frames_captioning(self, video_path):
        video, fps = prepare_video(video_path, 512, "cpu", normalize=False)
        # pick each frame for each second
        video = video[::fps]
        video_nd = np.transpose(video.numpy(), (0, 2, 3, 1)).astype(np.uint8)
        pil_images = [Image.fromarray(frame) for frame in video_nd]

        caption_results = []
        for i, image in enumerate(pil_images):
            # image.save(f"temp/{str(i).zfill(5)}.png")
            caption = self.image_captioning(
                image, prompt="This is a video frame describing that"
            )
            caption_results.append(f"Second {i}: {caption}.")
        return " ".join(caption_results)

    def inference(self, inputs):
        return self.frames_captioning(inputs)
