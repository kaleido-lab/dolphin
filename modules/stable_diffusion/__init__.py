from diffusers import StableDiffusionPipeline, PNDMScheduler
from utils import generate_image_name

class Text2Image:
    def __init__(self):
        self.pndm = PNDMScheduler.from_config("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        self.pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", scheduler=self.pndm)

    def image_generation(self, text):
        image = self.pipeline(text).images[0]
        image_url = generate_image_name()
        image.save(image_url)
        return image_url

