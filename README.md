# 高中物理备课工具

上传作业/试卷 PDF，自动切题、AI 分析知识点、生成讲评 PPT 和板书笔记。

## 功能

- **自动切题**：识别 PDF 中的题目边界，逐题裁切为图片
- **人工校对**：可手动分割、合并、删除识别错误的题目
- **AI 分析**（DeepSeek V3）：
  - 逐题给出知识点和解题思路
  - 按核心考点归类分组
  - 生成可直接上课用的板书（Markdown 格式，支持 KaTeX 数学公式）
- **一键导出**：讲评 PPT、板书 Markdown、分析 JSON

## 快速开始

**1. 安装依赖**

```bash
pip install -r requirements.txt
```

**2. 启动服务**

```bash
python app.py
```

或双击 `run.bat`

**3. 打开浏览器**

访问 [http://localhost:5000](http://localhost:5000)

**4. 使用流程**

1. 填入 DeepSeek API Key（[在此获取](https://platform.deepseek.com/)）
2. 上传作业/试卷 PDF（可选：附上课件 PDF 作为参考）
3. 等待自动切题，检查题目图片，按需调整
4. 点击「开始 AI 分析」
5. 下载 PPT 和板书

## 技术栈

| 组件 | 说明 |
|------|------|
| Flask | Web 后端 |
| RapidOCR + ONNX Runtime | 题目图片文字识别 |
| PyMuPDF | PDF 解析与渲染 |
| DeepSeek V3 API | 知识点分析与板书生成 |
| python-pptx | PPT 生成 |
| Pillow | 图像处理 |

## 系统要求

- Python 3.10+
- Windows / macOS / Linux

## 注意事项

- 需要自备 DeepSeek API Key，API Key 仅在本地使用，不会上传或存储
- 上传的 PDF 文件保存在本地 `uploads/` 目录，重启服务后历史任务会清空
- 扫描件 PDF（无文字图层）也支持，OCR 识别后送入 AI 分析

## License

MIT
