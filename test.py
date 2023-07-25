from modules.sadtalker import Sadtalker
import torch

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

sd = Sadtalker(device)
sd.inference("audio/ac9fc7da.wav,image/test.png")

