# TorchBlox 🧩

可视化 PyTorch 模型设计器 — 像搭积木一样构建神经网络。

## 🌐 在线体验

**[https://xz6663.dpdns.org/](https://xz6663.dpdns.org/)**

## ✨ 功能特性

- **拖拽式设计** — 从左侧模块库拖拽到画布，像搭积木一样构建模型
- **30+ 内置模块** — Linear、Conv2d、LSTM、Transformer、残差连接、池化、归一化等
- **实时形状推断** — 每步操作自动计算张量维度变化
- **一键导出代码** — 生成标准 PyTorch 代码（`__init__` + `forward`）
- **多格式导出** — 支持 `.py` / `.ipynb` / ONNX / TorchScript
- **AI 导入分析** — 粘贴 Python 代码自动分析并转换为积木图（需 DeepSeek API Key）
- **自定义模块** — 选中多个节点打包为可复用模块
- **模板一键加载** — 内置经典网络模板，也可保存自己的模板
- **撤销/重做** — 完整的历史记录支持
- **响应式布局** — 适配桌面端与移动端

## 🚀 快速开始

```bash
git clone https://github.com/xz6663/torchblox.git
cd torchblox
# 直接打开 app.html 或部署到任意静态服务器
```

## 📦 项目结构

```
torchblox/
├── app.html               # 主页面（PyTorch 可视化设计器）
├── vercel.json            # Vercel 部署配置
├── js/
│   ├── blocks.js          # 模块定义（30+ 个 PyTorch 模块）
│   ├── shape-inference.js # 维度推断引擎
│   ├── codegen.js         # PyTorch 代码生成
│   ├── custom.js          # 自定义模块 + 模板管理
│   ├── ai.js              # DeepSeek API 集成
│   ├── editor.js          # 画布交互引擎
│   └── app.js             # 主逻辑控制
├── _shared/
│   └── fonts/             # 字体文件
├── assets/                # 静态资源
└── test/                  # 测试文件
```

## 🖼️ 截图

（待补充）

## 🛠️ 技术栈

- 纯前端静态应用（无框架依赖）
- SVG + DOM 混合渲染
- Canvas 画布交互
- DeepSeek Chat API（AI 代码分析）
- localStorage（数据持久化）

## 📄 许可

MIT License
