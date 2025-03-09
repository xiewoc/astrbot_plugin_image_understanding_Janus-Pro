from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api.message_components import *
from astrbot.api.all import *
import os
import sys
import subprocess

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'Janus'))
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'Janus'))

global quantize, model

def run_command(command):#cmd line  git required!!!!
    import subprocess 
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    if error:
        print(f"Error: {error.decode()}")
    return output.decode()

def download_dependences():
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),'Janus')):
        pass
    else:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Janus')
        run_command(f"git clone --recursive https://github.com/deepseek-ai/Janus.git {base_dir}")

def is_video_file(filename):
    # 常见视频文件扩展名
    video_extensions = {
        '.avi', '.mp4', '.mov', '.wmv', '.mkv', '.flv', '.webm', '.vob',
        '.ogv', '.ogg', '.drc', '.mts', '.m2ts', '.ts', '.mxf', '.rm',
        '.asf', '.amv', '.m4v', '.mpg', '.mpeg', '.3gp', '.f4v', '.f4p',
        '.f4a', '.f4b'
    }
    
    # 提取文件的扩展名并转换为小写
    ext = '.' + filename.split('.')[-1].lower()
    
    # 检查扩展名是否在视频扩展名集合中  
    return ext in video_extensions

def quantize_model(model):
    from bitsandbytes import nn as bnn
    import torch
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # 使用bitsandbytes提供的线性层替代原始的线性层
            quantized_layer = bnn.Linear8bitLt(
                module.in_features, 
                module.out_features,
                module.bias is not None
            )
            # 将原layer的权重和偏置复制到新的量化层中
            quantized_layer.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                quantized_layer.bias.data.copy_(module.bias.data)
            # 替换原来的模块
            parent_name, attr_name = name.rsplit('.', 1)
            setattr(model.get_submodule(parent_name), attr_name, quantized_layer)
    return model

async def describe(file_path, model, if_quantize):
    from transformers import AutoModelForCausalLM
    from janus.models import MultiModalityCausalLM, VLChatProcessor
    from janus.utils.io import load_pil_images
    from modelscope import snapshot_download
    import torch

    # specify the path to the model
    model_name = 'deepseek-ai/' + model
    model_path = snapshot_download(model_name, local_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', model_name))
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    if if_quantize:#int8 q + bfloat16
        vl_gpt = quantize_model(vl_gpt)
        if torch.cuda.is_available():
            vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
        else:
            vl_gpt = vl_gpt.to("cpu").eval()
    else:
        if torch.cuda.is_available():
            vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
        else:
            vl_gpt = vl_gpt.to("cpu").eval()
    question = '描述一下这个图片里面有什么，尽量详细到人名、动作、表情、部位细节、物品名称、文字及位置等，尽量详细到人名'
    image = file_path
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    # load images and prepare for inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(vl_gpt.device)

    # # run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # # run the model to get the response
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    print("answer:",answer)
    return answer

@register("astrbot_plugin_image_understanding_Janus-Pro", "xiewoc", "为本地模型提供的图片理解补充，使用deepseek-ai/Janus-Pro系列模型", "1.0.1", "https://github.com/xiewoc/astrbot_plugin_image_understanding_Janus-Pro")
class astrbot_plugin_image_video_understanding_Janus_Pro(Star):
    def __init__(self, context: Context,config: dict):
        super().__init__(context)
        self.config = config
        
        global quantize, model
        quantize = self.config['enable_using_quantize']
        model = self.config['model']
        
    @event_message_type(EventMessageType.PRIVATE_MESSAGE)
    async def on_message(self, event: AstrMessageEvent):
        #print(event.message_obj.message) # AstrBot 解析出来的消息链内容
        global quantize ,input_text ,model
        input_text = ''
        if event.is_at_or_wake_command:
            opt = ''
            for item in event.message_obj.message:
                if isinstance(item, Image):#图像解析
                    if event.get_platform_name() == "aiocqhttp":
                        # qq
                        from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
                        assert isinstance(event, AiocqhttpMessageEvent)
                        client = event.bot # 得到 client
                        payloads = {
                            "file_id": item.file,
                            }
                        ret = await client.api.call_action('get_file', **payloads) # 调用协议端  API
                        #print(ret)
                        path = ret['file']
                        #print(path)
                        #save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'temp',ret['file'])
                        #print(save_path)
                        # 使用 curl
                        #subprocess.run(["curl", "-o", save_path, url])
                        opt = await describe(path,model,quantize)
                    else:
                        pass
                else:
                    opt = None
            for item in event.message_obj.message:
                if opt:
                    if isinstance(item, Plain):
                        input_text = item.text + '#图片内容：' + opt
                    else:
                        input_text = '#图片内容：' + opt
                else:
                    pass
        else:
            pass
    from astrbot.api.provider import ProviderRequest

    @filter.on_llm_request()
    async def on_call_llm(self, event: AstrMessageEvent, req: ProviderRequest): # 因为发送图片会有call_llm的动作，所以是在on_request的时候加
        global input_text
        if input_text !='':
            req.system_prompt += input_text
download_dependences()
