# IndexTTS 2.5 长文本小说配音（Colab）

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MAE5blog/indextts-2-5-novel-tts-colab/blob/main/IndexTTS_2_5_Novel_TTS_Colab.ipynb)

一个面向单旁白长文本的 Google Colab 工作流。它使用官方 IndexTTS 2.5 做推理，本仓库只负责中文文本切分、`manifest.json`、逐段原子保存、断点续跑与 WAV/M4B 合并。

这不是现有 VoxCPM2 项目的分支，也不复制其代码：两者的模型、实现和许可证彼此独立。

## 适用范围

- 仅使用已经获得授权的参考音频、书稿和目标声音。
- 免费 Colab 的 T4（16 GB）只能视为实验性路线：Notebook 固定 FP32、关闭 Qwen 情感模型，先通过“一段试听 + 连续十段”再跑长书。
- 不启动 Gradio、WebUI、share 链接或隧道；所有推理由 Python API 在当前 Colab runtime 内执行。
- 默认不跑全书；只有将 `CONFIRM_RUN_FULL_BOOK = True` 且使用 Google Drive 持久化目录后才会开始。

## 快速开始

1. 点击上方 **Open in Colab**，在 Colab 的“运行时 → 更改运行时类型”选择 GPU。
2. 运行 Notebook 的 0–5 节。第 4 节会在官方源码目录创建独立的 Python 3.11 环境；不要把 IndexTTS 依赖装进 Colab 内核的 Python 3.12。
3. 在 Google Drive 的 `MyDrive/IndexTTS_2_5_Novel/inputs/` 中放入已授权的参考音频和书稿。Notebook 不会调用上传对话框。
4. 在第 6 节设置文件名和任务名，依次运行下载、规划和试听。
5. 满意后反复运行第 9 节；每次最多生成十段，已完成片段会自动跳过。
6. 全部完成后运行第 10 节，Notebook 会合并并播放最终 WAV；需要 M4B 时只需将该格的 `EXPORT_M4B` 改为 `True`。

## 目录布局

```text
MyDrive/IndexTTS_2_5_Novel/
├── inputs/                       # 私有书稿和参考音频，不进 Git
└── jobs/<JOB_NAME>/
    ├── manifest.json             # 文本、参数、音频哈希和段状态
    ├── previews/smoke.wav
    ├── segments/s000001.wav      # 每段完成后立即落盘
    └── exports/audiobook.m4b
```

重新连接或中断后，重新运行 Notebook 的同步、存储、安装、配置、下载和“继续生成下一批”单元即可；已有且有效的片段会自动跳过。

## 本地检查

```powershell
python -m py_compile .\indextts25_novel.py
python -m unittest discover -s .\tests -v
python .\scripts\build_notebook.py
```

## 上游与许可证

- 官方代码：[index-tts/index-tts](https://github.com/index-tts/index-tts)
- 官方模型：[IndexTeam/IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5)
- 本项目自身代码：MIT，见 [LICENSE](LICENSE)
- IndexTTS 代码与模型：仍适用上游 Bilibili Model Use License，见 [UPSTREAM_NOTICES.md](UPSTREAM_NOTICES.md)

免费 Colab 的 GPU 类型、时长和可用性由平台动态分配，不能保证长篇生产。若 T4 加载失败或连续生成不稳定，请改用具备原生 BF16 且显存更高的 L4/4090/A100 级 GPU。
