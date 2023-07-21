import os
import wave
from utils import generate_audio_name
from modelscope.models.audio.tts import SambertHifigan
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


class BarkVoiceClone:
    
    def __init__(self):
        self.model_dir = os.path.abspath("./modules/bark_voice_clone/pretrain_work_dir")
        self.output_file = generate_audio_name()
        self.num_channels = 1
        self.sample_width = 2
        self.frame_rate = 18050 
    
    def inference(self, prompt):
        custom_infer_abs = {
            'voice_name':
            'F7',
            'am_ckpt': os.path.join(self.model_dir, 'tmp_am', 'ckpt'),
            'am_config': os.path.join(self.model_dir, 'tmp_am', 'config.yaml'),
            'voc_ckpt': os.path.join(self.model_dir, 'orig_model', 'basemodel_16k', 'hifigan', 'ckpt'),
            'voc_config': os.path.join(self.model_dir, 'orig_model', 'basemodel_16k', 'hifigan', 'config.yaml'),
            'audio_config': os.path.join(self.model_dir, 'data', 'audio_config.yaml'),
            'se_file': os.path.join(self.model_dir, 'data', 'se', 'se.npy')
        }
        kwargs = {'custom_ckpt': custom_infer_abs}

        model_id = SambertHifigan(os.path.join(self.model_dir, "orig_model"), **kwargs)

        inference = pipeline(task=Tasks.text_to_speech, model=model_id)
        output = inference(input=prompt)

        with wave.open(self.output_file, 'wb') as wav_file:
            wav_file.setnchannels(self.num_channels)
            wav_file.setsampwidth(self.sample_width)
            wav_file.setframerate(self.frame_rate)
            wav_file.writeframesraw(output["output_wav"])