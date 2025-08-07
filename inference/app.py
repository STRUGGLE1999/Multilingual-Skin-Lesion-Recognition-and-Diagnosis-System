import gradio as gr
from PIL import Image
import numpy as np
import mindspore as ms
from mindspore import Tensor, context, load_checkpoint, load_param_into_net
from mindcv.models import create_model  # 从 MindCV 导入模型构建方法
from openai import OpenAI
import base64
import io
import os

# 设置 MindSpore 运行模式
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# 初始化 Lingshu-32B Client
client = OpenAI(
    base_url="https://ai.gitee.com/v1",
    api_key="HHKM7PMBA4SKWMRZFQMLGZL73IMSUH8PVKF3FT1M",
)

# 标签映射（根据你的文件夹名称按字典序排序）
labels = {
    0: 'CHP',      # Chickenpox
    1: 'CWP',      # Cowpox
    2: 'HEALTHY',
    3: 'HFMD',
    4: 'MKP',      # MPOX
    5: 'MSL'       # Measles
}
label_name_map = {
    "CHP": "Chickenpox",
    "CWP": "Cowpox",
    "HEALTHY": "Healthy",
    "HFMD": "Hand,foot and mouth disease",
    "MKP": "MPOX",
    "MSL": "Measles"
}

# 加载 MindCV 的 ResNet50 模型
net = create_model(model_name='densenet121', num_classes=6, pretrained=False)
param_dict = load_checkpoint("densenet121_best.ckpt")
load_param_into_net(net, param_dict)
net.set_train(False)

# 图像预处理函数
def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image).astype(np.float32)
    if image_array.ndim == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    image_array = image_array.transpose(2, 0, 1) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return Tensor(image_array, ms.float32)

# 推理函数
def predict(image):
    input_tensor = preprocess_image(image)
    output = net(input_tensor)

    # 添加 softmax 将 logit 转换为概率
    softmax = ms.nn.Softmax()
    probs = softmax(output)

    prediction = probs.asnumpy()[0]
    top_index = np.argmax(prediction)
    confidence = prediction[top_index] * 100
    label_code = labels[top_index]
    return label_name_map[label_code], confidence


# 调用 Lingshu-32B 生成诊断报告
def generate_report(label, language="zh", image=None):
    lang_map = {
        "zh": "请用中文生成关于该皮肤病的诊断建议",
        "en": "Please write a medical explanation in English for the condition",
        "ar": "يرجى تقديم تشخيص طبي لهذا المرض الجلدي باللغة العربية"
    }

    prompt = f"病种：{label}。\n患者上传了一张皮肤病照片。请基于图像和病种，为医生提供初步诊断建议，包括症状描述、可能病因、是否需要就医、注意事项。\n{lang_map.get(language)}"

    image_url = None
    if image:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{img_base64}"

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

    return response.choices[0].message.content.strip()

# Gradio 界面交互函数
def gradio_interface(image, language):
    if image is None:
        return "请上传图片", ""
    label, confidence = predict(image)
    result_text = f"模型识别为：**{label}**\n置信度：{confidence:.2f}%"
    report = generate_report(label, language=language, image=image)
    return result_text, report

# 示例图片及对应标签，请替换为你的真实路径和标签
example_images = [
    ["../examples/CHP_07_01.jpg", "Chickenpox"],
    ["../examples/CWP_03_01.jpg", "Cowpox"],
    ["../examples/HEALTHY_105_01.jpg", "Healthy"],
    ["../examples/HFMD_03_01.jpg", "Hand,foot and mouth disease"],
    ["../examples/MKP_06_01.jpg", "Mpox"],
    ["../examples/MSL_11_01.jpg", "Measles"]
]

# 加载 PIL 图像列表
example_imgs_pil = []
for path, label in example_images:
    if os.path.exists(path):
        example_imgs_pil.append(Image.open(path))
    else:
        example_imgs_pil.append(Image.new("RGB", (224, 224), (200, 200, 200)))

# 构建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>🧪 多语言皮肤病分类与诊断系统（DenseNet121）</h1>")
    gr.Markdown("<p style='text-align: center;'>上传图像，自动识别皮肤病，并生成多语种诊断建议</p>")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="📤 上传或拍摄皮肤图像")
            language_input = gr.Radio(
                choices=["zh", "en", "ar"],
                label="🌍 选择诊断语言",
                value="zh"
            )
            submit_button = gr.Button("🔍 开始诊断")
        with gr.Column():
            top_output = gr.Textbox(label="🔬 输出识别结果", lines=3)
            report_output = gr.Textbox(label="📝 自动生成诊断报告", lines=8)
    
    submit_button.click(fn=gradio_interface,
                        inputs=[image_input, language_input],
                        outputs=[top_output, report_output])

    # 示例图像静态展示（2行3列，每张图带标签）
    gr.Markdown("### 🎯 示例图像快速测试（仅展示）")
    
    gallery = gr.Gallery(label="示例图像", show_label=False, value=example_imgs_pil, columns=3, height="auto")

    # 点击示例图像 -> 显示到 image_input 框
    def on_gallery_select(evt: gr.SelectData):
        return example_imgs_pil[evt.index]

    gallery.select(fn=on_gallery_select, inputs=[], outputs=[image_input])

    # with gr.Row():
    #     for i in range(2):  # 第一行 3 张图
    #         with gr.Column():
    #             img = gr.Image(value=example_images[i][0], show_label=False, interactive=True,height=150)
    #             img.click(fn=lambda idx=i: update_image_input(idx), inputs=[], outputs=[image_input])
    #             gr.Markdown(f"<center>**{example_images[i][1]}**</center>")

    # with gr.Row():
    #     for i in range(2, 4):  # 第二行 3 张图
    #         with gr.Column():
    #             img = gr.Image(value=example_images[i][0], show_label=False, interactive=True,height=150)
    #             img.click(fn=lambda idx=i: update_image_input(idx), inputs=[], outputs=[image_input])
    #             gr.Markdown(f"<center>**{example_images[i][1]}**</center>")
    # with gr.Row():
    #     for i in range(4, 6):  # 第二行 3 张图
    #         with gr.Column():
    #             img = gr.Image(value=example_images[i][0], show_label=False, interactive=True,height=150)
    #             img.click(fn=lambda idx=i: update_image_input(idx), inputs=[], outputs=[image_input])
    #             gr.Markdown(f"<center>**{example_images[i][1]}**</center>")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
