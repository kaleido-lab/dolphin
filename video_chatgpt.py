import argparse
import re, uuid, os, shutil
from omegaconf import OmegaConf

import gradio as gr

from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI

import utils


VIDEO_CHATGPT_PREFIX = """Video ChatGPT is designed to be able to assist with a wide range of text and visual related tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. Video ChatGPT is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Video ChatGPT is able to process and understand large amounts of text and videos. As a language model, Video ChatGPT can not directly read videos, but it has a list of tools to finish different visual tasks. Each video will have a file name formed as "video/xxx.mp4", and Video ChatGPT can invoke different tools to indirectly understand videos. When talking about videos, Video ChatGPT is very strict to the file name and will never fabricate nonexistent files. When using tools to generate new video files, Video ChatGPT is also known that the video may not be the same as the user's demand, and will use other visual question answering tools or description tools to observe the real video. Video ChatGPT is able to use tools in a sequence, and is loyal to the tool observation outputs rather than faking the video content and video file name. It will remember to provide the file name from the last tool observation, if a new video is generated.

Human may provide new videos to Video ChatGPT with a description. The description helps Video ChatGPT to understand this video, but Video ChatGPT should use tools to finish following tasks, rather than directly imagine from the description.

Overall, Video ChatGPT is a powerful visual dialogue assistant tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 


TOOLS:
------

Video ChatGPT  has access to the following tools:"""

VIDEO_CHATGPT_FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""

VIDEO_CHATGPT_SUFFIX = """You are very strict to the filename correctness and will never fake a file name if it does not exist.
You will remember to provide the v file name loyally if it's provided in the last tool observation.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
Since Video ChatGPT is a text language model, Video ChatGPT must use tools to observe videos rather than imagination.
The thoughts and observations are only visible for Video ChatGPT, Video ChatGPT should remember to repeat important information in the final response for Human. 
Thought: Do I need to use a tool? {agent_scratchpad}"""


VIDEO_CHATGPT_PREFIX_CN = """Video ChatGPT 旨在能够协助完成范围广泛的文本和视觉相关任务，从回答简单的问题到提供对广泛主题的深入解释和讨论。 Video ChatGPT 能够根据收到的输入生成类似人类的文本，使其能够进行听起来自然的对话，并提供连贯且与手头主题相关的响应。

Video ChatGPT 能够处理和理解大量文本和视频。作为一种语言模型，Video ChatGPT 不能直接读取视频，但它有一系列工具来完成不同的视觉任务。每个视频都会有一个文件名，格式为“video/xxx.mp4”，Video ChatGPT可以调用不同的工具来间接理解视频。在谈论视频时，Video ChatGPT 对文件名的要求非常严格，绝不会伪造不存在的文件。在使用工具生成新的视频文件时，Video ChatGPT也知道视频可能与用户需求不一样，会使用其他视觉问答工具或描述工具来观察真实视频。 Video ChatGPT 能够按顺序使用工具，并且忠于工具观察输出，而不是伪造视频内容和视频文件名。如果生成新视频，它将记得提供上次工具观察的文件名。

Human 可能会向 Video ChatGPT 提供带有描述的新视频。描述帮助 Video ChatGPT 理解这个视频，但 Video ChatGPT 应该使用工具来完成以下任务，而不是直接从描述中想象。有些工具将会返回英文描述，但你对用户的聊天应当采用中文。

总的来说，Video ChatGPT 是一个强大的可视化对话辅助工具，可以帮助处理范围广泛的任务，并提供关于范围广泛的主题的有价值的见解和信息。

工具列表:
------

Video ChatGPT 可以使用这些工具:"""

VIDEO_CHATGPT_FORMAT_INSTRUCTIONS_CN = """用户使用中文和你进行聊天，但是工具的参数应当使用英文。如果要调用工具，你必须遵循如下格式:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

当你不再需要继续调用工具，而是对观察结果进行总结回复时，你必须使用如下格式：


```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""

VIDEO_CHATGPT_SUFFIX_CN = """你对文件名的正确性非常严格，而且永远不会伪造不存在的文件。

开始!

因为Video ChatGPT是一个文本语言模型，必须使用工具去观察视频而不是依靠想象。
推理想法和观察结果只对Video ChatGPT可见，需要记得在最终回复时把重要的信息重复给用户，你只能给用户返回中文句子。我们一步一步思考。在你使用工具时，工具的参数只能是英文。

聊天历史:
{chat_history}

新输入: {input}
Thought: Do I need to use a tool? {agent_scratchpad}
"""


os.makedirs("video", exist_ok=True)
os.makedirs("image", exist_ok=True)
os.makedirs("audio", exist_ok=True)
os.makedirs("modeling", exist_ok=True)

def cut_dialogue_history(history_memory, keep_last_n_words=500):
    if history_memory is None or len(history_memory) == 0:
        return history_memory
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f"history_memory:{history_memory}, n_tokens: {n_tokens}")
    if n_tokens < keep_last_n_words:
        return history_memory
    paragraphs = history_memory.split("\n")
    last_n_tokens = n_tokens
    while last_n_tokens >= keep_last_n_words:
        last_n_tokens -= len(paragraphs[0].split(" "))
        paragraphs = paragraphs[1:]
    return "\n" + "\n".join(paragraphs)


class ConversationBot:
    def __init__(self, cfg: OmegaConf):
        print(f"Initializing VideoChatGPT, load_cfg={cfg}")

        # instantiate model zoos
        for k, v in cfg.model_zoos.items():
            print("k:", k, "v:", v)
        self.models = {
            k: utils.instantiate_from_config(v) for k, v in cfg.model_zoos.items()
        }

        # instantiate tools
        self.tools = [
            Tool(
                name=v.name,
                description=v.desc,
                func=getattr(self.models[v.instance], v.func),
            )
            for v in cfg.tools
            if v.instance in self.models
        ]

        self.memory = ConversationBufferMemory(
            memory_key="chat_history", output_key="output"
        )

    def init_agent(self, openai_api_key, lang):
        self.memory.clear()  # clear previous history
        if lang == "English":
            PREFIX, FORMAT_INSTRUCTIONS, SUFFIX = (
                VIDEO_CHATGPT_PREFIX,
                VIDEO_CHATGPT_FORMAT_INSTRUCTIONS,
                VIDEO_CHATGPT_SUFFIX,
            )
            place = "Enter text and press enter, or upload a video"
            label_clear = "Clear"
        else:
            PREFIX, FORMAT_INSTRUCTIONS, SUFFIX = (
                VIDEO_CHATGPT_PREFIX_CN,
                VIDEO_CHATGPT_FORMAT_INSTRUCTIONS_CN,
                VIDEO_CHATGPT_SUFFIX_CN,
            )
            place = "输入文字并回车，或者上传视频"
            label_clear = "清除"
        self.llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={
                "prefix": PREFIX,
                "format_instructions": FORMAT_INSTRUCTIONS,
                "suffix": SUFFIX,
            },
        )
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(placeholder=place),
            gr.update(value=label_clear),
        )

    def run_text(self, text, state):
        self.agent.memory.buffer = cut_dialogue_history(
            self.agent.memory.buffer, keep_last_n_words=500
        )
        res = self.agent({"input": text})
        res["output"] = res["output"].replace("\\", "/")

        response = re.sub(
            "(video/[-\w]*.mp4)",
            lambda m: f"<video controls autoplay width='320'><source id='mp4' src='file={m.group(0)}' type='video/mp4'></video>*{m.group(0)}*",
            res["output"],
        )
        response = re.sub(
            "(video/[-\w]*.wav)",
            lambda m: f"<audio controls autoplay><source id='wav' src='file={m.group(0)}' type='audio/mpeg'></audio>*{m.group(0)}*",
            response,
        )
        state = state + [(text, response)]
        print(
            f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
            f"Current Memory: {self.agent.memory.buffer}"
        )
        return state, state

    def run_video(self, video, state, txt, lang):
        video_filename = os.path.join("video", f"{str(uuid.uuid4())[:8]}.mp4")

        shutil.move(video.name, video_filename)

        description = self.models["VideoCaptioning"].inference(video_filename)
        frames_caption = self.models["ImageCaptioning"].inference(video_filename)

        if lang == "Chinese":
            Human_prompt = f'\nHuman: 提供一个名为 {video_filename}的视频。它的描述是: {description}。它的每一秒画面的描述是：{frames_caption}。这些信息帮助你理解这个视频，如果询问你一些关于这个视频内容的问题，你可以从我的描述中总结和想象，但对于视频编辑、处理、生成的任务，请使用工具来完成。如果你明白了, 说 "收到". \n'
            AI_prompt = "收到。  "
        else:
            Human_prompt = f'\nHuman: provide a video named {video_filename}. The description is: {description}. The caption of each second is: {frames_caption}. This information helps you to understand this video. If you are asked questions about the content of this video, you can summarize and imagine from my description. But for tasks of video editing, processing or generating, please use tools to finish following tasks, rather than directly imagine. If you understand, say "Received". \n'
            AI_prompt = "Received.  "

        self.agent.memory.buffer = (
            self.agent.memory.buffer + Human_prompt + "AI: " + AI_prompt
        )
        # state = state + [(f"![](/file={video_filename})*{video_filename}*", AI_prompt)]
        state = state + [
            (
                f"<video controls autoplay width='320'><source id='mp4' src='file={video_filename}' type='video/mp4'></video>*{video_filename}*",
                AI_prompt,
            )
        ]
        print(
            f"\nProcessed run_video, Input video: {video_filename}\nCurrent state: {state}\n"
            f"Current Memory: {self.agent.memory.buffer}"
        )
        return state, state, f"{txt} {video_filename} "


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./configs/backends.yaml")
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--port", type=int, default=7890)
    return parser


def main():
    # command line arguments
    parser = get_parser()
    args, _ = parser.parse_known_args()

    # initialize
    cfg = OmegaConf.load(args.cfg)
    if args.load is not None:
        # only load user customized model
        load_dict = {
            e.split("_")[0].strip(): e.split("_")[1].strip()
            for e in args.load.split(",")
        }
        
        diff_set = set(load_dict.keys()).difference(set(dict(cfg.model_zoos).keys()))
        if len(diff_set) > 0:
            raise ValueError(
                f"Key {diff_set} not in model_zoos. Please check the config file `configs/backends.yaml`"
            )

        for key in dict(cfg.model_zoos).keys():
            
            if key in load_dict:
                if "params" in cfg.model_zoos[key]:
                    cfg.model_zoos[key].params.device = load_dict[key]
            
            else:
                del cfg.model_zoos[key]
                
                to_dels = [
                    i for i, e in enumerate(list(cfg.tools)) if e.instance == key
                ]
                
                for index in sorted(to_dels, reverse=True):
                    del cfg.tools[index]

    bot = ConversationBot(cfg)

    # gradio frontend
    with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
        gr.Markdown("<h3><center>Dolphin: Talk to Video</center></h3>")
        gr.Markdown(
            """**Dolphin** is a general video interaction platform based on large language models. See our [Project](https://github.com/kaleido-lab/dolphin).
            """
        )

        with gr.Row():
            lang = gr.Radio(
                choices=["Chinese", "English"], value="English", label="Language"
            )
            openai_api_key_textbox = gr.Textbox(
                placeholder="Paste your OpenAI API key here to start Dolphin(sk-...) and press Enter ↵️",
                show_label=False,
                lines=1,
                type="password",
            )

        chatbot = gr.Chatbot(elem_id="chatbot", label="dolphin")
        state = gr.State([])

        with gr.Row(visible=False) as input_raws:
            with gr.Column(scale=0.7):
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="Enter text and press enter, or upload a video",
                ).style(container=False)
            with gr.Column(scale=0.10, min_width=0):
                run = gr.Button("🏃‍♂️ Run")
            with gr.Column(scale=0.15, min_width=0):
                clear = gr.Button("🔄 Clear")
            with gr.Column(scale=0.15, min_width=0):
                btn = gr.UploadButton("🎥 Upload", file_types=["video"])

        gr.Examples(
            examples=[
                "Generate a video describing a goldendoodle playing in a park by a lake",
                "Make the video snowy",
                "Can you detect the depth video of this video?",
                "Can you use this pose video to generate an astronaut?",
                "Describe this video",
                "Replace the dog with a cat",
                "Make it water-color painting",
                "What clothes is the person wearing?",
                "Please detect the pose of this video",
                "What is the background?",
            ],
            inputs=txt,
        )

        openai_api_key_textbox.submit(
            bot.init_agent,
            [openai_api_key_textbox, lang],
            [input_raws, lang, txt, clear],
        )
        txt.submit(bot.run_text, [txt, state], [chatbot, state])
        txt.submit(lambda: "", None, txt)
        run.click(bot.run_text, [txt, state], [chatbot, state])
        run.click(lambda: "", None, txt)
        btn.upload(bot.run_video, [btn, state, txt, lang], [chatbot, state, txt])
        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)
    demo.queue(concurrency_count=10).launch(
        server_name="0.0.0.0", server_port=args.port, share=True
    )


if __name__ == "__main__":
    main()
