---
license: apache-2.0
---

## 基于 MindSpore 和 Lingshu-32B 的皮肤病智能识别与多语言诊断系统
### 项目简介
本项目构建了一个皮肤病图像识别与诊断报告自动生成系统，集成图像分类与大语言模型，支持中文、英文、阿拉伯语三种语言，适用于基层医疗辅助、远程皮肤病筛查和医学教育场景。

系统实现了以下功能：

- 使用 MindSpore 框架训练的轻量级 CNN 模型（LeNet）对 6 类皮肤病图像进行分类（MPOX、HFMD、Measles、Chickenpox、Cowpox、Healthy）。

- 调用 Lingshu-32B 多模态大语言模型，基于图像与分类标签自动生成医学诊断建议。

- 用户可选择中文、英文、阿拉伯语等输出语言，提升全球可用性与多语种医疗辅助能力。

- 提供 Gradio 界面，支持图像上传、模型预测与报告生成，一键部署。

### 技术栈
- 深度学习框架：MindSpore

- 语言大模型 API：Gitee · Lingshu-32B

- 前端交互：Gradio

- 语言支持：简体中文、英文、阿拉伯语

### 数据集说明
本项目使用 MSLD v2.0（Mpox Skin Lesion Dataset），包含以下6类皮肤病图像，共755张样本：

| 类别         | 样本数 |
|--------------|--------|
| Mpox         | 284    |
| Chickenpox   | 75     |
| Measles      | 55     |
| Cowpox       | 66     |
| HFMD         | 161    |
| Healthy      | 114    |

图像已处理为 224×224 尺寸 RGB 格式。

### 缩写说明（中文注释）：
MSLD：Monkeypox Skin Lesion Dataset，即猴痘皮肤病变图像数据集。

Mpox：Monkeypox（猴痘），一种由猴痘病毒引起的病毒性皮肤病，常见症状为发热、皮疹和淋巴结肿大。

HFMD：Hand-Foot-Mouth Disease（手足口病），一种常见于儿童的病毒感染病，特点是手、足、口腔出现水疱或溃疡。

Chickenpox：水痘，由水痘-带状疱疹病毒（VZV）引起，典型症状为全身皮疹和瘙痒。

Measles：麻疹，由麻疹病毒引起的高度传染性疾病，常见症状包括发热、咳嗽、眼结膜炎和皮疹。

Cowpox：牛痘，一种由牛痘病毒感染引起的皮肤病，历史上用于制造天花疫苗。

Healthy：健康个体，不含任何皮肤病变，用于对照研究。

### Gradio应用链接

https://xihe.mindspore.cn/projects/STRUGGLE/SkinLesionApp/
