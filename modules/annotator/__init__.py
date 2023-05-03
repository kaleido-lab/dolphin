import cv2
import torch
import numpy as np
from einops import rearrange
from torchvision.transforms import Resize, InterpolationMode

from .util import HWC3
from .openpose import OpenposeDetector
from .midas import MidasDetector

from utils import get_new_video_name
from video_utils import prepare_video, create_video


class Video2Canny:
    def __init__(self, **kwargs):
        print("Initializing Video2Canny")

    def pre_process_canny(self, input_video, low_threshold=100, high_threshold=200):
        detected_maps = []
        for frame in input_video:
            img = rearrange(frame, "c h w -> h w c").cpu().numpy().astype(np.uint8)
            detected_map = cv2.Canny(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)
            detected_maps.append(detected_map[None])
        detected_maps = np.concatenate(detected_maps)
        control = torch.from_numpy(detected_maps.copy()).float() / 255.0
        return rearrange(control, "f h w c -> f c h w")

    def inference(self, inputs):
        vid_path = inputs
        video, fps = prepare_video(vid_path, resolution=512, device="cpu")
        vid_canny = self.pre_process_canny(video)
        canny_to_save = list(
            rearrange(vid_canny, "f c w h -> f w h c").cpu().detach().numpy()
        )
        out_path = get_new_video_name(vid_path, "edge")
        return create_video(canny_to_save, fps, out_path)


class Video2Pose:
    def __init__(self, device, dtype=torch.float16):
        print("Initializing Video2Pose")
        self.device = device
        self.dtype = dtype
        self.detector = OpenposeDetector(device=device)

    def pre_process_pose(self, input_video, apply_pose_detect: bool = True):
        detected_maps = []
        for frame in input_video:
            img = rearrange(frame, "c h w -> h w c").cpu().numpy().astype(np.uint8)
            img = HWC3(img)
            if apply_pose_detect:
                detected_map, _ = self.detector(img)
            else:
                detected_map = img
            detected_map = HWC3(detected_map)
            H, W, C = img.shape
            detected_map = cv2.resize(
                detected_map, (W, H), interpolation=cv2.INTER_NEAREST
            )
            detected_maps.append(detected_map[None])
        detected_maps = np.concatenate(detected_maps)
        control = torch.from_numpy(detected_maps.copy()).float() / 255.0
        return rearrange(control, "f h w c -> f c h w")

    def inference(self, inputs, resolution=512):
        vid_path = inputs
        video, fps = prepare_video(
            vid_path, resolution=resolution, device=self.device, normalize=False
        )
        vid_pose = self.pre_process_pose(video)
        canny_to_save = list(
            rearrange(vid_pose, "f c w h -> f w h c").cpu().detach().numpy()
        )
        out_path = get_new_video_name(vid_path, "pose")
        return create_video(canny_to_save, fps, out_path)


class Video2Depth:
    def __init__(self, device, dtype=torch.float16):
        print("Initializing Video2Depth")
        self.device = device
        self.dtype = dtype
        self.depth_estimator = MidasDetector(device)

    def pre_process_depth(self, input_video, apply_depth_detect: bool = True):
        detected_maps = []
        for frame in input_video:
            img = rearrange(frame, "c h w -> h w c").cpu().numpy().astype(np.uint8)
            img = HWC3(img)
            if apply_depth_detect:
                detected_map, _ = self.depth_estimator(img)
            else:
                detected_map = img
            detected_map = HWC3(detected_map)
            H, W, C = img.shape
            detected_map = cv2.resize(
                detected_map, (W, H), interpolation=cv2.INTER_NEAREST
            )
            detected_maps.append(detected_map[None])
        detected_maps = np.concatenate(detected_maps)
        control = torch.from_numpy(detected_maps.copy()).float() / 255.0
        return rearrange(control, "f h w c -> f c h w")

    def inference(self, inputs, resolution=512):
        vid_path = inputs
        video, fps = prepare_video(
            vid_path,
            resolution=resolution,
            device=self.device,
            dtype=self.dtype,
            normalize=False,
        )
        control = self.pre_process_depth(video).to(self.device).to(self.dtype)

        depth_map_to_save = list(
            rearrange(control, "f c w h -> f w h c").cpu().detach().numpy()
        )
        out_path = get_new_video_name(vid_path, "depth")
        return create_video(depth_map_to_save, fps, out_path)
