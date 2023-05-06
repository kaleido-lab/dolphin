<p align="center" width="100%">
<img src="https://user-images.githubusercontent.com/78398294/236116868-00801805-5cbf-40d1-89a1-f848a15b1deb.png"  width="80%" height="80%">
</p>

<div>
<div align="center">
    Zehuan Huang&emsp;
    Haoran Feng&emsp;
    Chongzhi Zhang
    </br>
    <a href='https://lucassheng.github.io/' target='_blank'>Lu Sheng</a>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu</a>&emsp;
    <a href='https://amandajshao.github.io/' target='_blank'>Jing Shao</a>
</div>
<div>
<div align="center">
    Beihang University, Nanyang Technological University
</div>

# dolphin

<!-- <a src="https://img.shields.io/badge/%F0%9F%A4%97-Open%20in%20Spaces-blue" href="" target="_blank">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97-Open%20in%20Spaces-blue" alt="Open in Huggingface">
</a>
<a src="https://img.shields.io/badge/GPU%20Demo-Open-green?logo=alibabacloud" href="" target="_blank">
    <img src="https://img.shields.io/badge/GPU%20Demo-Open-green?logo=alibabacloud"> 
</a> -->
<a src="https://img.shields.io/twitter/follow/kaleido_lab?style=social" href="https://twitter.com/kaleido_lab" target="_blank">
  <img src="https://img.shields.io/twitter/follow/kaleido_lab?style=social">
</a>

Dolphin is a general video interaction platform based on large language models. Our team is trying to build a chatbot for video understanding, processing and generation.

We are continuously improving 🐬 dolphin. Stay tuned for updates!

> Online demo is coming soon!

## 📽️ Demo

<a href="https://www.youtube.com/watch?v=d8giiMAWMLc" target="_blank">
  <img src="https://user-images.githubusercontent.com/78398294/236398299-050439e4-b870-44f5-8341-db87ca849748.png" alt="Dolphin, a general video interaction platform based on LLMs, from BUAA & NTU">
</a>

## 🔥 Updates

- 2023/05/06: Code release & Online Demo

  - Video understanding: Q&A about the video.
  - Video processing: Basic functions such as trimming video, adding subtitles, extracting audio, and adding audio using [moviepy](https://github.com/Zulko/moviepy). Video to pose/depth/canny also included.
  - Video generation: Text to video, pose/depth and text to video, and video pix2pix.

## 💬 Example

https://user-images.githubusercontent.com/78398294/236603247-b7381154-743c-4262-ad30-75f11e34a91d.mp4

## 🔨 Quick Start

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

To start 🐬 dolphin, you can specify the GPU/CPU assignment by `--load`, the parameter indicates which Video Foundation Model to use and where it will be loaded to. The model and device are separated by underline `_`, while the different models are separated by comma `,`. The available Video Foundation Models can be found in the following table or `configs/backends.yaml`.

For example, if you want to load VideoCaptioning to cuda:0 and MoviepyInterface to cpu, you can use: `VideoCaptioning_cuda:0,MoviepyInterface_cpu`.

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

## 💾 GPU memory usage

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

## 🛠️ How to expand

Our project framework is highly extensible for adding new features, including support for more video foundation models and more large language models.

For more video foundation models, you can add the inference code for new models under the `modules` directory. We recommend creating a new Python package for the new model within this directory and implementing the class in the package's `__init__.py` file (you can refer to `ModelscopeT2V` in `modules/modelscope_t2v/init.py`). Afterward, add the related information in `configs/backends.yaml`.

<details>
  <summary><b>Example: FaceText2Video</b></summary>

Assuming you have now implemented a new video foundation model using facial keypoints and text descriptions to generate videos, follow these steps:

1. Create a new package named `face2video` under the directory `modules`. In its `__init__.py` file, implement a class called `FaceText2Video`, which includes an initialization and an inference function. The desired effect should be that by importing `modules.face2video.FaceText2Video`, after instantiating an object, you can use the `inference` function to achieve the corresponding functionality.
2. Add the import and function description in `configs/backends.yaml`. Details are as follows.

```yaml
model_zoos:
  FaceText2Video:  # foundation model class
    target: modules.face2video.FaceText2Video # path of the class in project
    params: # params passed to the class
      device: cuda:0
tools:
  # - name: tool name
  #   desc: description about new tool
  #   instance: keep the name consistent with the one in the model_zoos section mentioned above
  #   func: inference function in foundation model class
  - name: Generate Video Condition On Face Video
    desc: "useful when you want to generate a new video from both the user description and a facial keypoints video. /
      like: generate a new video of a human face from this human face video, or can you generate a video based on both the text 'A boy is playing basketball.' and this face video. /
      The input to this tool should be a comma separated string of two, representing the video_path and the user description. "
    instance: FaceText2Video
    func: inference
```

</details>

For more large language models, you can refer to `video_chatgpt.py`, and create a new file like `video_moss.py` or `video_stablelm.py` in the project root directory to implement support for other large language models.

## ⏳ Ongoing

- [x] Chatbot with video downstream works (video understanding, processing and generation)
- [ ] Pretrained unified video model with in-context learning
- [ ] Benchmark for emerging video tasks
- [ ] Service including Gradio, Web, API and Docker

## 🤝 Acknowledgement

We appreciate the open source of the following projects:

[Hugging Face](https://github.com/huggingface) &#8194;
[LangChain](https://github.com/hwchase17/langchain) &#8194;
[mPLUG](https://github.com/alibaba/AliceMind/tree/main/mPLUG) &#8194; 
[BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) &#8194; 
[MoviePy](https://github.com/Zulko/moviepy) &#8194; 
[Text2Video-Zero](https://github.com/Picsart-AI-Research/Text2Video-Zero) &#8194;
[damo/text-to-video-synthesis](https://modelscope.cn/models/damo/text-to-video-synthesis/summary) &#8194;
[bark](https://github.com/suno-ai/bark)

## 📩 Contact Information

For help or issues using the 🐬 dolphin, please submit a GitHub issue.

For other communications, please contact Zehuan Huang (huanngzh@gmail.com) or kaleido lab (kaleido.ailab@gmail.com). Welcome to follow us in Twitter (<a href="https://twitter.com/kaleido_lab" target="_blank">@kaleido_lab</a>).

## 📎 Citation

If you find this repository useful, please consider citing:

```
@misc{stable-dreamfusion,
    Author = {Zehuan Huang, Haoran Feng, Enshen Zhou, Jiahua Lan, Chongzhi Zhang},
    Year = {2023},
    Note = {https://github.com/kaleido-lab/dolphin},
    Title = {Dolphin: General Video Interaction Platform Based on LLMs}
}
```