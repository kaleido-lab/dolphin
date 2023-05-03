import torch

from .model import (
    CannyText2VideoModel,
    PoseText2VideoModel,
    DepthText2VideoModel,
    VideoPix2PixModel,
    Text2VideoModel,
)

from utils import generate_video_name_mp4, get_new_video_name


class CannyText2Video:
    def __init__(self, device):
        self.device = device
        self.model = CannyText2VideoModel(device, dtype=torch.float16)

    def inference(self, inputs: str, resolution=512):
        vid_path, prompt = inputs.split(",")[0], ",".join(inputs.split(",")[1:])
        out_path = get_new_video_name(vid_path, func_name="canny2video")
        self.model.process_controlnet_canny(
            vid_path,
            prompt,
            save_path=out_path,
            resolution=resolution,
        )
        return out_path


class PoseText2Video:
    def __init__(self, device):
        self.device = device
        self.model = PoseText2VideoModel(device, dtype=torch.float16)

    def inference(self, inputs: str, resolution=512):
        vid_path, prompt = inputs.split(",")[0], ",".join(inputs.split(",")[1:])
        out_path = get_new_video_name(vid_path, func_name="pose2video")
        self.model.process_controlnet_pose(
            vid_path,
            prompt,
            save_path=out_path,
            resolution=resolution,
        )
        return out_path


class DepthText2Video:
    def __init__(self, device):
        self.device = device
        self.model = DepthText2VideoModel(device, dtype=torch.float16)

    def inference(self, inputs: str, resolution=512):
        vid_path, prompt = inputs.split(",")[0], ",".join(inputs.split(",")[1:])
        out_path = get_new_video_name(vid_path, func_name="depth2video")
        self.model.process_controlnet_depth(
            vid_path,
            prompt,
            save_path=out_path,
            resolution=resolution,
        )
        return out_path


class VideoPix2Pix:
    def __init__(self, device):
        self.device = device
        self.model = VideoPix2PixModel(device, dtype=torch.float16)

    def inference(self, inputs: str):
        vid_path, prompt = inputs.split(",")[0], ",".join(inputs.split(",")[1:])
        out_path = get_new_video_name(vid_path, func_name="pix2pix")
        self.model.process_pix2pix(
            vid_path,
            prompt,
            save_path=out_path,
        )
        return out_path


class Text2Video:
    def __init__(self, device):
        self.device = device
        self.model = Text2VideoModel(device, dtype=torch.float16)

    def inference(self, inputs: str, resolution=512):
        prompt = inputs
        params = {
            "t0": 44,
            "t1": 47,
            "motion_field_strength_x": 12,
            "motion_field_strength_y": 12,
            "video_length": 16,
        }
        out_path, fps = generate_video_name_mp4(), 8
        self.model.process_text2video(
            prompt,
            fps=fps,
            path=out_path,
            resolution=resolution,
            **params,
        )
        return out_path
