import imageio
import torch
import numpy as np
import decord
import torchvision
from einops import rearrange
from torchvision.transforms import Resize, InterpolationMode

from utils import get_new_video_name


def prepare_video(
    video_path: str,
    resolution: int,
    device,
    dtype=torch.float16,
    normalize=True,
    start_t: float = 0,
    end_t: float = -1,
    output_fps: int = -1,
):
    vr = decord.VideoReader(video_path)
    initial_fps = vr.get_avg_fps()
    if output_fps == -1:
        output_fps = int(initial_fps)
    if end_t == -1:
        end_t = len(vr) / initial_fps
    else:
        end_t = min(len(vr) / initial_fps, end_t)
    assert 0 <= start_t < end_t
    assert output_fps > 0
    start_f_ind = int(start_t * initial_fps)
    end_f_ind = int(end_t * initial_fps)
    num_f = int((end_t - start_t) * output_fps)
    sample_idx = np.linspace(start_f_ind, end_f_ind, num_f, endpoint=False).astype(int)
    video = vr.get_batch(sample_idx)
    if torch.is_tensor(video):
        video = video.detach().cpu().numpy()
    else:
        video = video.asnumpy()
    _, h, w, _ = video.shape
    video = rearrange(video, "f h w c -> f c h w")
    video = torch.Tensor(video).to(device).to(dtype)

    # Use max if you want the larger side to be equal to resolution (e.g. 512)
    # k = float(resolution) / min(h, w)
    k = float(resolution) / max(h, w)
    h *= k
    w *= k
    h = int(np.round(h / 64.0)) * 64
    w = int(np.round(w / 64.0)) * 64

    video = Resize((h, w), interpolation=InterpolationMode.BILINEAR, antialias=True)(
        video
    )
    if normalize:
        video = video / 127.5 - 1.0
    return video, output_fps  # video: f c h w


def create_video(frames, fps, path, rescale=False):
    # frames: f h w c
    outputs = []
    for _, x in enumerate(frames):
        x = torchvision.utils.make_grid(torch.Tensor(x), nrow=4)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    imageio.mimsave(path, outputs, fps=fps)
    return path


def preprocess_video(video_path, out_path=None):
    if out_path is None:
        out_path = get_new_video_name(video_path, func_name="preprocessed")

    video, fps = prepare_video(video_path, resolution=512, device="cpu")
    video = rearrange(video, "f c h w -> f h w c")
    create_video(video, fps, out_path, rescale=True)
    print(f"Preprocessed video saved to {out_path}")
