import gradio as gr
from PIL import Image
import numpy as np
import mindspore as ms
from mindspore import Tensor, context
import mindspore.nn as nn
from mindspore.train.serialization import load
from openai import OpenAI
import requests
import base64
import io


client = OpenAI(
    base_url="https://ai.gitee.com/v1",
    api_key="HHKM7PMBA4SKWMRZFQMLGZL73IMSUH8PVKF3FT1M",  # 你已提供
)

# 设置MindSpore上下文
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

def generate_report(label, language="zh", image=None):
    # 构造提示词
    lang_map = {
        "zh": "请用中文生成关于该皮肤病的诊断建议",
        "en": "Please write a medical explanation in English for the condition",
        "ar": "يرجى تقديم تشخيص طبي لهذا المرض الجلدي باللغة العربية"
    }

    prompt = f"病种：{label}。\n患者上传了一张皮肤病照片。请基于图像和病种，为医生提供初步诊断建议，包括症状描述、可能病因、是否需要就医、注意事项。\n{lang_map.get(language)}"

    # 如果传入了图像（PIL对象），我们需要将其 base64 编码
    image_url = None
    if image:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{img_base64}"

    # 构建消息体
    messages = [
        {
            "role": "system",
            "content": "You are a helpful and harmless assistant. You should think step-by-step."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    # 执行 Lingshu-32B 推理
    response = client.chat.completions.create(
        messages=messages,
        model="Lingshu-32B",
        stream=False,
        max_tokens=512,
        temperature=0.7,
        top_p=1,
        extra_body={"top_k": -1},
        frequency_penalty=0,
    )

    # 提取结果
    return response.choices[0].message.content.strip()

# 定义模型结构（和你之前的一样）
class LeNet(nn.Cell):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # 自动计算 flatten 后的大小
        dummy_input = Tensor(np.zeros((1, 3, 224, 224)), ms.float32)
        x = self.pool(self.relu(self.conv1(dummy_input)))
        x = self.pool(self.relu(self.conv2(x)))
        self.flatten_size = x.shape[1] * x.shape[2] * x.shape[3]

        self.fc1 = nn.Dense(self.flatten_size, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 6)

    def construct(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 加载 mindir 模型
net = LeNet()
load("best.mindir", net=net)

labels = {
    0: 'Chickenpox',
    1: 'Cowpox',
    2: 'HFMD',
    3: 'Healthy',
    4: 'Measles',
    5: 'MPOX'
}

def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image).astype(np.float32)
    if image_array.ndim == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    image_array = image_array.transpose(2, 0, 1) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return Tensor(image_array, ms.float32)

def predict(image):
    input_tensor = preprocess_image(image)
    output = net(input_tensor)
    prediction = output.asnumpy()[0]
    top_indices = prediction.argsort()[-3:][::-1]
    top_classes = [f"{labels[i]} ({prediction[i]*100:.2f}%)" for i in top_indices]
    return labels[top_indices[0]], "\n".join(top_classes)  # 返回Top-1标签 和 Top-3字符串

# Gradio界面函数
def gradio_interface(image, language):
    if image is None:
        return "请上传图片", ""
    top1_label, top3_text = predict(image)
    report = generate_report(top1_label, language=language, image=image)
    return top3_text, report

# 构建Gradio界面
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Image(type="pil", label="上传或拍摄皮肤图像"),
        gr.Radio(choices=["zh", "en", "ar"], label="输出语言", value="zh")
    ],
    outputs=[
        gr.Textbox(label="分类结果"),
        gr.Textbox(label="自动生成诊断报告")
    ],
    title="皮肤病分类 + 多语种诊断系统",
    description="上传皮肤图像，识别病种，并自动生成中文、英文或阿拉伯语诊断建议（Lingshu-32B 大模型）"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)

