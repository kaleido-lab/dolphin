import os, sys, uuid
import importlib

import numpy as np
import torch
import random


def instantiate_from_config(config, **kwargs):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()), **kwargs)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def get_new_video_name(org_vid_name, func_name="update"):
    head_tail = os.path.split(org_vid_name)
    head = head_tail[0]
    tail = head_tail[1]
    name_split = tail.split(".")[0].split("_")
    this_new_uuid = str(uuid.uuid4())[:4]
    if len(name_split) == 1:
        most_org_file_name = name_split[0]
    else:
        assert len(name_split) == 4
        most_org_file_name = name_split[3]
    recent_prev_file_name = name_split[0]
    new_file_name = (
        f"{this_new_uuid}_{func_name}_{recent_prev_file_name}_{most_org_file_name}.mp4"
    )
    return os.path.join(head, new_file_name)


def generate_video_name_mp4():
    return os.path.join("video", str(uuid.uuid4())[:8] + ".mp4")

def generate_audio_name():
    return os.path.join("audio", str(uuid.uuid4())[:8] + ".wav")

def generate_image_name():
    return os.path.join("image", str(uuid.uuid4())[:8] + ".png")

def get_new_uuid():
    return str(uuid.uuid4())[:8]
