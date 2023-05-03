from .get_video_caption import prepare_model, pipeline


mplug_model_zoo = "model_zoo/mplug"


class VideoCaptioning:
    def __init__(self, device):
        print("Initializing mPLUG for VideoCaptioning")
        self.download_models()
        self.device = device
        self.model, self.tokenizer = prepare_model(device)
        self.pipe = pipeline

    def inference(self, inputs):
        return pipeline(inputs, self.model, self.tokenizer, self.device)

    def download_models(self):
        model_list = [
            "https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/ViT-L-14.tar",
            "https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/mPLUG/mplug_large.pth",
        ]
        for url in model_list:
            from basicsr.utils.download_util import load_file_from_url

            load_file_from_url(url, model_dir=mplug_model_zoo)
