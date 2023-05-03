import os
from moviepy.editor import *

from utils import get_new_video_name, generate_audio_name


def is_chinese_or_english(string):
    for char in string:
        if "\u4e00" <= char <= "\u9fff":
            return "cn"
        elif ("\u0041" <= char <= "\u005a") or ("\u0061" <= char <= "\u007a"):
            return "en"
    return "en"


class MoviepyInterface:
    def __init__(self, **kwargs):
        print("Initializing MoviepyInterface")

    def intercept_fragments(self, inputs):
        print(inputs)
        splits = inputs.split(",")
        image_path, begin, end = (
            splits[0],
            float(splits[1].strip()),
            float(splits[2].strip()),
        )

        clip = VideoFileClip(image_path).subclip(begin, end)
        newclip = get_new_video_name(inputs, func_name="subclip")
        clip.write_videofile(newclip)
        return newclip

    def add_subtitles(self, inputs, color="white"):
        try:
            video_path, start_time, duration, instruct_text = (
                inputs.split(",")[0],
                float(inputs.split(",")[1].strip()),
                float(inputs.split(",")[2].strip()),
                ",".join(inputs.split(",")[3:]).strip(),
            )
        except:
            video_path, instruct_text = (
                inputs.split(",")[0],
                ",".join(inputs.split(",")[1:]).strip(),
            )
            start_time = 0
            duration = None

        def get_font():
            if is_chinese_or_english(instruct_text) == "cn":
                return os.path.join(os.path.dirname(__file__), "font", "cn.ttf")
            else:
                return "Amiri-Bold"

        clip = VideoFileClip(video_path)
        duration = duration if duration else clip.duration
        w, h = clip.w, clip.h
        txt_clip = (
            TextClip(
                instruct_text,
                fontsize=h / 10,
                font=get_font(),
                size=(w - 20, h / 10),
                align="center",
                color=color,
            )
            .set_position((10, h - h / 6))
            .set_duration(duration)
            .set_start(start_time)
        )
        video = CompositeVideoClip([clip, txt_clip])
        newvideo = get_new_video_name(inputs, func_name="subtitles")
        video.write_videofile(newvideo)
        return newvideo

    def concat_videos(self, inputs, resolution=512):
        splits = inputs.split(",")
        clips = [VideoFileClip(split.strip()) for split in splits]
        w, h = clips[0].w, clips[0].h
        ratio = w / h
        clips = [clip.resize((int(resolution * ratio), resolution)) for clip in clips]
        new_video = get_new_video_name(splits[0], func_name="video-concat")
        concatenate_videoclips(clips).write_videofile(new_video)
        return new_video

    def video_overlay(self, inputs):
        splits = inputs.split(",")
        clips = [VideoFileClip(split) for split in splits]
        new_video = get_new_video_name(splits[0], func_name="overlay-video")
        clips_array(clips).write_videofile(new_video)
        return new_video

    def extract_audio(self, inputs):
        video_path = inputs
        audio_path = generate_audio_name()
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_path)
        return audio_path

    def add_audio_to_video(self, inputs):
        video_path, audio_path = (
            inputs.split(",")[0].strip(),
            inputs.split(",")[1].strip(),
        )
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)
        duration = min(video_clip.duration, audio_clip.duration)
        video_clip = video_clip.set_duration(duration)
        final_video = video_clip.set_audio(audio_clip.set_duration(duration))
        new_video_path = get_new_video_name(video_path, func_name="add-audio")
        final_video.write_videofile(new_video_path)
        return new_video_path
