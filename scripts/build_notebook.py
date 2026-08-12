"""Regenerate the Colab notebook from its scaffold with stable, reviewable cells."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "IndexTTS_2_5_Novel_TTS_Colab.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).lstrip().splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).lstrip().splitlines(keepends=True),
    }


cells = [
    markdown(
        """
        # IndexTTS 2.5 单旁白小说配音（Google Colab）

        这是一个**有授权声音与书稿**的长文本工作流：官方 IndexTTS 2.5 负责推理，本仓库负责分段、`manifest.json` 断点续跑、单段原子落盘和最终合并。它不会启动 WebUI、隧道或公网服务，也不会弹出浏览器上传窗口。

        **免费 Colab / T4 仅作实验性验证。** T4 没有原生 BF16，本 Notebook 固定 FP32（`use_bf16=False`）并关闭 Qwen 情感模型。先完成“模型加载 + 一段试听 + 连续十段”再决定是否跑全书。
        """
    ),
    markdown(
        """
        ## 流程与边界

        1. 克隆本仓库并运行本地逻辑测试。
        2. 选择 Google Drive（正式任务）或 `/content`（仅临时试听）。
        3. 用隔离 Python 3.11 安装官方固定版本的 IndexTTS；不把它混装进 Colab 的 Python 3.12 内核。
        4. 直接调用官方 Python API：FP32、禁用 QwenEmotion、禁用 DeepSpeed/CUDA kernel。
        5. 用 `manifest.json` 逐段保存；中断后重跑“生成全书”会跳过已验证的 WAV。

        不提交参考音频、书稿、模型、任务目录或成品。只使用你拥有或已获明确授权的声音和文字。
        """
    ),
    markdown("## 0. 获取或同步本仓库代码"),
    code(
        """
        from pathlib import Path
        import os
        import shutil
        import subprocess
        import sys

        REPO_URL = "https://github.com/MAE5blog/indextts-2-5-novel-tts-colab.git"
        REPO_REF = "main"
        REPO_DIR = Path("/content/indextts-2-5-novel-tts-colab")

        def run(command, *, cwd=None, env=None):
            print("+", " ".join(map(str, command)))
            return subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True)

        if (REPO_DIR / ".git").is_dir():
            run(["git", "fetch", "origin", REPO_REF], cwd=REPO_DIR)
            run(["git", "checkout", "-f", REPO_REF], cwd=REPO_DIR)
            run(["git", "reset", "--hard", f"origin/{REPO_REF}"], cwd=REPO_DIR)
        else:
            shutil.rmtree(REPO_DIR, ignore_errors=True)
            run(["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, REPO_DIR])

        sys.path.insert(0, str(REPO_DIR))
        print("项目代码：", REPO_DIR)
        """
    ),
    markdown("## 1. 先检查仓库自身的分段与断点逻辑（无需 GPU）"),
    code(
        """
        run([sys.executable, "-m", "unittest", "discover", "-s", REPO_DIR / "tests", "-v"], cwd=REPO_DIR)
        """
    ),
    markdown("## 2. Colab GPU 健康检查"),
    code(
        """
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("未检测到 GPU：请在 Colab 的“运行时 → 更改运行时类型”中选择 GPU 后重新连接。")

        props = torch.cuda.get_device_properties(0)
        capability = torch.cuda.get_device_capability(0)
        print({
            "name": props.name,
            "vram_gib": round(props.total_memory / 1024**3, 2),
            "compute_capability": f"{capability[0]}.{capability[1]}",
        })
        if capability[0] < 8:
            print("警告：这是 T4 或更早的 GPU。IndexTTS 2.5 将使用 FP32；只先做短样本验证。")
        """
    ),
    markdown("## 3. 选择任务存储位置"),
    code(
        """
        from pathlib import Path

        # 正式任务保持 drive；content 只适合手工放入 /content 后的单段试听，运行时结束会清空。
        STORAGE_MODE = "drive"  # "drive" 或 "content"
        DRIVE_ROOT = "MyDrive/IndexTTS_2_5_Novel"

        if STORAGE_MODE == "drive":
            from google.colab import drive
            drive.mount("/content/drive", force_remount=False)
            STORAGE_ROOT = Path("/content/drive") / DRIVE_ROOT
        elif STORAGE_MODE == "content":
            STORAGE_ROOT = Path("/content/IndexTTS_2_5_Novel")
            print("临时模式：不要运行整本生成；所有内容会随运行时结束而丢失。")
        else:
            raise ValueError("STORAGE_MODE 只能是 'drive' 或 'content'")

        INPUTS_DIR = STORAGE_ROOT / "inputs"
        JOBS_DIR = STORAGE_ROOT / "jobs"
        INPUTS_DIR.mkdir(parents=True, exist_ok=True)
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        print("输入目录：", INPUTS_DIR)
        print("任务目录：", JOBS_DIR)
        """
    ),
    markdown(
        """
        ## 4. 安装官方 IndexTTS 2.5（隔离 Python 3.11）

        上游要求 Python `<3.12`；此单元创建官方源码自己的 Python 3.11 虚拟环境。安装和首次权重下载较大，可能需要十多分钟。不要把 IndexTTS 的 PyTorch 包安装到当前 Colab 内核。
        """
    ),
    code(
        """
        # 官方代码与权重固定到本次验证时的提交；需要更新时先在仓库中评审后再改这里。
        INDEXTTS_COMMIT = "b5ea881bec284b72f0b1cc04e0a724ff0c6b93e9"
        INDEXTTS_MODEL_ID = "IndexTeam/IndexTTS-2.5"
        INDEXTTS_MODEL_REVISION = "ba2480d9f7f629eb18f6acaebb357679d9ba88a4"
        UPSTREAM_DIR = Path("/content/index-tts")

        run(["sudo", "apt-get", "-qq", "update"])
        run(["sudo", "apt-get", "-qq", "install", "-y", "ffmpeg"])
        run([sys.executable, "-m", "pip", "install", "-q", "uv"])
        UV = shutil.which("uv") or str(Path(sys.executable).parent / "uv")

        if not (UPSTREAM_DIR / ".git").is_dir():
            shutil.rmtree(UPSTREAM_DIR, ignore_errors=True)
            run(["git", "clone", "https://github.com/index-tts/index-tts.git", UPSTREAM_DIR])
        run(["git", "fetch", "origin", INDEXTTS_COMMIT], cwd=UPSTREAM_DIR)
        run(["git", "checkout", "--detach", INDEXTTS_COMMIT], cwd=UPSTREAM_DIR)
        run([UV, "python", "install", "3.11"], cwd=UPSTREAM_DIR)
        # 不使用 --all-extras：DeepSpeed / flash-attn 在免费 Colab 中没有必要且更易出错。
        run([UV, "sync", "--python", "3.11", "--extra", "webui"], cwd=UPSTREAM_DIR)

        VENV_PY = UPSTREAM_DIR / ".venv" / "bin" / "python"
        if not VENV_PY.is_file():
            raise RuntimeError(f"找不到 IndexTTS Python 环境：{VENV_PY}")
        print("IndexTTS Python：", VENV_PY)
        """
    ),
    markdown("## 5. 运行时帮助函数（从隔离 Python 3.11 调用项目命令）"),
    code(
        """
        import json

        CLI = REPO_DIR / "indextts25_novel.py"
        MODEL_DIR = Path("/content/checkpoints_25")  # 运行时本地磁盘更快；重连后可重新下载。

        def worker(*args, capture_output=False):
            env = os.environ.copy()
            inherited = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = os.pathsep.join([str(REPO_DIR), str(UPSTREAM_DIR), inherited])
            command = [str(VENV_PY), str(CLI), *map(str, args)]
            return subprocess.run(command, env=env, check=True, text=True, capture_output=capture_output)

        def job_summary():
            manifest_path = JOB_DIR / "manifest.json"
            if not manifest_path.is_file():
                print("尚未建立任务清单。请先运行第 7 节。")
                return None
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            counts = {}
            for segment in manifest["segments"]:
                status = segment.get("status", "unknown")
                counts[status] = counts.get(status, 0) + 1
            summary = {
                "状态": manifest.get("status"),
                "总段数": len(manifest["segments"]),
                "已完成": counts.get("completed", 0),
                "待生成": counts.get("pending", 0),
                "失败": counts.get("failed", 0),
                "任务目录": str(JOB_DIR),
            }
            print(summary)
            return summary

        preflight = worker("preflight", capture_output=True)
        print(preflight.stdout)
        print("生成器已就绪。")
        """
    ),
    markdown("## 6. 输入与任务配置（只改这一格）"),
    code(
        """
        # storage 用 Drive 文件；inline 可将下方文字保存到 Drive；repo_demo 仅试跑原创短文本。
        BOOK_SOURCE = "storage"  # "storage"、"inline" 或 "repo_demo"
        BOOK_FILE_NAME = "chapter01_background_test.txt"  # storage / inline 时使用
        REFERENCE_AUDIO_FILE = "古龙评书（干声）.flac"  # 你的授权参考音频，放在 INPUTS_DIR
        JOB_NAME = "history_ch01_background_t4_20260812"
        LANG = "ZH"

        # 仅 BOOK_SOURCE="inline" 时使用；可直接粘贴短文本或一个章节。
        BOOK_TEXT = \\"\\"\"
        夜里刚下过一阵小雨，窗外的梧桐叶还挂着水珠。
        \\"\\"\"

        # 这些参数会写入 manifest；改动后请改 JOB_NAME，避免混用旧成品。
        TARGET_CHARS = 60
        HARD_CHARS = 80
        MAX_TEXT_TOKENS = 80

        if BOOK_SOURCE == "repo_demo":
            BOOK_PATH = REPO_DIR / "examples" / "original_demo.txt"
        elif BOOK_SOURCE == "storage":
            BOOK_PATH = INPUTS_DIR / BOOK_FILE_NAME
        elif BOOK_SOURCE == "inline":
            if not BOOK_TEXT.strip():
                raise ValueError("BOOK_TEXT 为空：请粘贴需要生成的文字。")
            BOOK_PATH = INPUTS_DIR / BOOK_FILE_NAME
            if BOOK_PATH.suffix.lower() not in {".txt", ".md", ".markdown"}:
                raise ValueError("BOOK_FILE_NAME 请使用 .txt、.md 或 .markdown 后缀。")
            BOOK_PATH.write_text(BOOK_TEXT.strip() + "\\n", encoding="utf-8")
            print("已将粘贴文本保存到：", BOOK_PATH)
        else:
            raise ValueError("BOOK_SOURCE 只能是 'storage'、'inline' 或 'repo_demo'")
        REFERENCE_AUDIO_PATH = INPUTS_DIR / REFERENCE_AUDIO_FILE
        JOB_DIR = JOBS_DIR / JOB_NAME
        if not BOOK_PATH.is_file():
            raise FileNotFoundError(f"找不到书稿：{BOOK_PATH}")
        if not REFERENCE_AUDIO_PATH.is_file():
            raise FileNotFoundError(f"找不到参考音频：{REFERENCE_AUDIO_PATH}")
        print({"book": str(BOOK_PATH), "reference_audio": str(REFERENCE_AUDIO_PATH), "job": str(JOB_DIR)})
        """
    ),
    markdown("## 7. 下载权重并创建可续跑任务清单"),
    code(
        """
        # checkpoint 约 5 GiB，首次下载会较久；该命令可安全重复执行。
        worker("download", "--model-dir", MODEL_DIR, "--model-id", INDEXTTS_MODEL_ID, "--revision", INDEXTTS_MODEL_REVISION)
        worker(
            "plan",
            "--job-dir", JOB_DIR,
            "--text", BOOK_PATH,
            "--reference-audio", REFERENCE_AUDIO_PATH,
            "--model-id", INDEXTTS_MODEL_ID,
            "--model-revision", INDEXTTS_MODEL_REVISION,
            "--target-chars", TARGET_CHARS,
            "--hard-chars", HARD_CHARS,
            "--max-tokens", MAX_TEXT_TOKENS,
        )
        print("任务清单已就绪；同名、同配置任务会保留已生成的片段。")
        job_summary()
        """
    ),
    markdown("## 8. 必做：先试听一段"),
    code(
        """
        from IPython.display import Audio, display

        SMOKE_TEXT = "夜里刚下过一阵小雨，窗外的梧桐叶还挂着水珠。"
        SMOKE_WAV = JOB_DIR / "previews" / "smoke.wav"
        worker(
            "smoke",
            "--model-dir", MODEL_DIR,
            "--reference-audio", REFERENCE_AUDIO_PATH,
            "--output", SMOKE_WAV,
            "--text", SMOKE_TEXT,
            "--lang", LANG,
        )
        print("试听已生成：", SMOKE_WAV)
        display(Audio(str(SMOKE_WAV)))
        """
    ),
    markdown("## 9. 继续生成下一批（每次最多 10 段，可反复运行）"),
    code(
        """
        # 自动跳过已完成的片段，从下一个待生成片段继续。
        # 只要 Drive 已挂载，断连后重跑 0、3–7、9 即可继续。
        BATCH_SIZE = 10
        RETRY_FAILED_SEGMENTS = False
        render_args = [
            "render",
            "--job-dir", JOB_DIR,
            "--model-dir", MODEL_DIR,
            "--reference-audio", REFERENCE_AUDIO_PATH,
            "--lang", LANG,
            "--limit", BATCH_SIZE,
        ]
        if RETRY_FAILED_SEGMENTS:
            render_args.append("--retry-failed")
        worker(*render_args)
        summary = job_summary()
        if summary and summary["待生成"] == 0 and summary["失败"] == 0:
            print("全部片段已完成，可以运行第 10 节合并。")
        else:
            print("还未全部完成时，直接再次运行本单元格即可。")
        """
    ),
    markdown("## 10. 合并、导出并播放成品（全部完成后运行）"),
    code(
        """
        from IPython.display import Audio, display

        EXPORT_M4B = False
        summary = job_summary()
        if summary is None or summary["待生成"] or summary["失败"]:
            print("尚未全部生成完毕：请继续运行第 9 节。")
        else:
            exports = JOB_DIR / "exports"
            wav_path = exports / f"{JOB_NAME}.wav"
            merge_args = [
                "merge",
                "--job-dir", JOB_DIR,
                "--wav", wav_path,
            ]
            if EXPORT_M4B:
                merge_args += ["--m4b", exports / f"{JOB_NAME}.m4b"]
            worker(
                *merge_args,
            )
            print("成品已导出到：", exports)
            display(Audio(str(wav_path)))
        """
    ),
    markdown(
        """
        ## 恢复与常见问题

        - **Colab 断连或 GPU 回收**：重新运行 0、3、4、5、6、7、9。已有的有效 WAV 会被跳过。
        - **T4 在模型加载时 OOM**：这是免费 T4 的实验性边界，不要改成 BF16 或自行伪造 FP16；换到 L4/4090/A100 级 GPU 后再试。
        - **某一段失败**：第 9 节会显示失败数。确认是临时问题后，把其中的 `RETRY_FAILED_SEGMENTS` 改为 `True` 再运行；若书稿或参数有调整，请换 `JOB_NAME` 后从第 7 节重建。
        - **没有声音 / 失真**：保持官方锁定的 Python 3.11、Torch 2.8 环境；不要把 Colab 内核的 Torch 版本混入上游环境。
        """
    ),
    markdown(
        """
        ## 上游与许可

        - [IndexTTS 官方仓库](https://github.com/index-tts/index-tts)
        - [IndexTTS-2.5 官方权重](https://huggingface.co/IndexTeam/IndexTTS-2.5)
        - 本仓库仅提供 MIT 许可的编排代码；IndexTTS 代码和模型仍适用其上游条款，详见 `UPSTREAM_NOTICES.md`。
        """
    ),
]


notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
notebook["cells"] = cells
notebook["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
    "accelerator": "GPU",
    "colab": {"provenance": []},
}
notebook["nbformat"] = 4
notebook["nbformat_minor"] = 5
NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {NOTEBOOK}")
