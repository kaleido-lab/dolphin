import ruamel.yaml as yaml
import numpy as np
import torch
import torch.nn as nn
from .models.model_caption_mplug_vatex import MPLUG
from .models.vit import interpolate_pos_embed, resize_pos_embed
from .models.tokenization_bert import BertTokenizer
from decord import VideoReader
import decord
import os


config_path = os.path.join("model_zoo", "mplug", "videocap_vatex_mplug_large.yaml")
mplug_pth_path = os.path.join("model_zoo", "mplug", "mplug_large.pth")

config = yaml.load(open(config_path, "r"), Loader=yaml.Loader)


def prepare_model(device):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = MPLUG(config=config, tokenizer=tokenizer)
    model = model.to(device)

    assert os.path.exists(
        mplug_pth_path
    ), "Please download mplug_large.pth checkpoint from https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/mplug_large.pth and put it in ./model_zoo/mplug/"
    checkpoint = torch.load(mplug_pth_path, map_location=device)

    try:
        state_dict = checkpoint["model"]
    except:
        state_dict = checkpoint["module"]
    if config["clip_name"] == "ViT-B-16":
        num_patches = int(config["image_res"] * config["image_res"] / (16 * 16))
    elif config["clip_name"] == "ViT-L-14":
        num_patches = int(config["image_res"] * config["image_res"] / (14 * 14))

    pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768).float())
    pos_embed = resize_pos_embed(
        state_dict["visual_encoder.visual.positional_embedding"].unsqueeze(0),
        pos_embed.unsqueeze(0),
    )
    state_dict["visual_encoder.visual.positional_embedding"] = pos_embed

    for key in list(state_dict.keys()):
        if ("fusion" in key or "bert" in key) and "decode" not in key:
            encoder_key = key.replace("fusion.", "").replace("bert.", "")
            state_dict[encoder_key] = state_dict[key]
            del state_dict[key]

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, tokenizer


def pipeline(video_path, model, tokenizer, device):
    video = load_video_from_path_decord(
        video_path, config["image_res"], config["image_res"], config["num_frm_test"]
    ).to(device)
    if config["prompt"] != "":
        caption = [config["prompt"] + config["eos"]] * video.size(0)
        caption = tokenizer(
            caption,
            padding="longest",
            truncation=True,
            max_length=25,
            return_tensors="pt",
        ).to(device)
    else:
        caption = None

    topk_ids, topk_probs = model(video, caption, None, train=False)

    for topk_id, topk_prob in zip(topk_ids, topk_probs):
        ans = (
            tokenizer.decode(topk_id[0])
            .replace("[SEP]", "")
            .replace("[CLS]", "")
            .replace("[PAD]", "")
            .strip()
        )
        ans += " ."
        return ans


def load_video_from_path_decord(
    video_path,
    height=None,
    width=None,
    num_frame=12,
    start_time=None,
    end_time=None,
    fps=-1,
):
    decord.bridge.set_bridge("torch")

    if not height or not width:
        vr = VideoReader(video_path)
    else:
        vr = VideoReader(video_path, width=width, height=height)
    vlen = len(vr)
    if start_time or end_time:
        assert fps > 0, "must provide video fps if specifying start and end time."
        start_idx = min(int(start_time * fps), vlen)
        end_idx = min(int(end_time * fps), vlen)
    else:
        start_idx, end_idx = 0, vlen

    frame_index = np.arange(start_idx, end_idx, vlen / num_frame, dtype=int)
    raw_sample_frms = vr.get_batch(frame_index)
    raw_sample_frms = raw_sample_frms.permute(0, 3, 1, 2).float().unsqueeze(0)

    return raw_sample_frms
