# dolphin

> TODO: page header

---

> TODO: online demo

<a src="https://img.shields.io/badge/%F0%9F%A4%97-Open%20in%20Spaces-blue" href="">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97-Open%20in%20Spaces-blue" alt="Open in Huggingface">
</a>
<a src="https://img.shields.io/badge/GPU%20Demo-Open-green?logo=alibabacloud" href="">
    <img src="https://img.shields.io/badge/GPU%20Demo-Open-green?logo=alibabacloud"> 
</a>

Dolphin is a general video interaction platform based on ChatGPT. Our team is trying to build a chatbot for video understanding, processing and generation.

We are continuously improving _dolphin_. Stay tuned for updates!

## Demo

> TODO: Video

## Updates

- 2023/05/04: Code release & Online Demo
  - Video understanding: Q&A about the video.
  - Video processing: Basic functions such as trimming video, adding subtitles, extracting audio, and adding audio using [moviepy](https://github.com/Zulko/moviepy). Video to pose/depth/canny also included.
  - Video generation: Text to video, pose/depth and text to video, and video pix2pix.

## Example

> TODO: example video

## Quick Start

Prepare the project and environment:

```Bash
# We recommend using conda to manage the environment and use python 3.8
conda create -n dolphin python=3.8
conda activate dolphin

# Clone the respository:
git clone https://github.com/BUAA-PrismGroup/dolphin.git
cd dolphin

# Install dependencies:
pip install -r requirements.txt
```

To start _dolphin_, you can specify the GPU/CPU assignment by "--load", the parameter indicates which Video Foundation Model to use and where it will be loaded to. The model and device are separated by underline '_', while the different models are separated by comma ','. The available Video Foundation Models can be found in the following table or _configs/backends.yaml_.

For example, if you want to load VideoCaptioning to cuda:0 and MoviepyInterface to cpu, you can use: "VideoCaptioning_cuda:0,MoviepyInterface_cpu".

Some starting commands are as follows.

```Bash
# Advice for CPU Users
python video_chatgpt.py --load VideoCaptioning_cpu,ImageCaptioning_cpu,ModelscopeT2V_cpu

# Advice for 1 Tesla T4 15GB
python video_chatgpt.py --load "VideoCaptioning_cuda:0,ImageCaptioning_cuda:0,ModelscopeT2V_cuda:0"

# Advice for 4 Tesla V100 32GB (Full usage)
# You can specify the device where each model is loaded in `configs/backend.yaml`
python video_chatgpt.py
```

## GPU memory usage

| **Foundation Model** | **GPU Memory (MB)** |
| - | - |
| VideoCaptioning | 13393 |
| ImageCaptioning | 8429 |
| MoviepyInterface | 0 |
| Video2Canny | 0 |
| Video2Pose | 1361 |
| Video2Depth | 1521 |
| CannyText2Video | 6659 |
| PoseText2Video | 6721 |
| DepthText2Video | 6673 |
| VideoPix2Pix | 5251 |
| ModelscopeT2V | 6535 |
| Text2Audio | 5797 |

## Acknowledgement

We appreciate the open source of the following projects:

[Hugging Face](https://github.com/huggingface) &#8194;
[LangChain](https://github.com/hwchase17/langchain) &#8194;
[mPLUG](https://github.com/alibaba/AliceMind/tree/main/mPLUG) &#8194; 
[BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) &#8194; 
[MoviePy](https://github.com/Zulko/moviepy) &#8194; 
[Text2Video-Zero](https://github.com/Picsart-AI-Research/Text2Video-Zero) &#8194;
[damo/text-to-video-synthesis](https://modelscope.cn/models/damo/text-to-video-synthesis/summary) &#8194;
[bark](https://github.com/suno-ai/bark) &#8194;

## Contact Information

For help or issues using the _dolphin_, please submit a GitHub issue.

> TODO: For other communications, please contact
