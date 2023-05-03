from scipy.io.wavfile import write as write_wav
from bark import SAMPLE_RATE, generate_audio, preload_models

from utils import generate_audio_name


class Text2Audio:
    def __init__(self, **kwargs):
        print("Initializing Bark for Text2Audio")
        # download and load all models
        print("Loading bark models for text2audio...")
        preload_models()

    def text2audio(self, inputs):
        # generate audio from text
        text = inputs
        audio_array = generate_audio(text)
        audio_path = generate_audio_name()
        write_wav(audio_path, SAMPLE_RATE, audio_array)
        return audio_path

    def text2music(self, inputs):
        # generate music from text
        text = "♪ " + inputs + " ♪"
        audio_array = generate_audio(text)
        audio_path = generate_audio_name()
        write_wav(audio_path, SAMPLE_RATE, audio_array)
        return audio_path
