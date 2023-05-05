import os
import textwrap
from moviepy.editor import *

from utils import get_new_video_name, generate_audio_name


def is_chinese_or_english(string):
    for char in string:
        if "\u4e00" <= char <= "\u9fff":
            return "cn"
        elif ("\u0041" <= char <= "\u005a") or ("\u0061" <= char <= "\u007a"):
            return "en"
    return "en"


def wrap_text_to_fit_video(text, video_width, fontsize, times=0.6):
    max_chars_per_line = int(video_width / (fontsize * times))
    wrapped_text = textwrap.fill(text, max_chars_per_line)
    return wrapped_text.split("\n")


def add_subtitles_to_video(
    video_path,
    text,
    fontsize=None,
    text_color="white",
    shadow_color="black",
    shadow_offset=(2, 2),
    start_time=0,
    duration=None,
):
    def get_font():
        if is_chinese_or_english(text) == "cn":
            return os.path.join(os.path.dirname(__file__), "font", "cn.ttf")
        else:
            return "Amiri-Bold"

    video = VideoFileClip(video_path)
    video_width, video_height = video.size
    duration = video.duration if duration is None else duration

    if fontsize is None:
        fontsize = video_height // 12
    lines = wrap_text_to_fit_video(
        text,
        video_width,
        fontsize,
        times=0.6 if is_chinese_or_english(text) == "en" else 1.2,
    )

    text_clips = []
    for idx, line in enumerate(lines):
        shadow_clip = (
            TextClip(line, fontsize=fontsize, color=shadow_color, font=get_font())
            .set_position(
                (
                    "center",
                    video_height
                    - (len(lines) - idx + 1 / 2) * fontsize
                    + shadow_offset[1],
                )
            )
            .set_duration(duration)
            .set_start(start_time)
        )
        text_clip = (
            TextClip(line, fontsize=fontsize, color=text_color, font=get_font())
            .set_position(
                ("center", video_height - (len(lines) - idx + 1 / 2) * fontsize)
            )
            .set_duration(duration)
            .set_start(start_time)
        )
        text_clips.extend([shadow_clip, text_clip])

    video_with_subtitles = CompositeVideoClip([video] + text_clips)

    return video_with_subtitles


class MoviepyInterface:
    def __init__(self, **kwargs):
        print("Initializing MoviepyInterface")

    def intercept_fragments(self, inputs):
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

    def add_subtitles(self, inputs):
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

        video = add_subtitles_to_video(
            video_path, instruct_text, start_time=start_time, duration=duration
        )
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
        final_video.write_videofile(new_video_path, codec="libx264", audio_codec="aac")
        return new_video_path
