# 运行命令

- 项目：NVIDIA NeMo Speech（ASR / TTS / Speech LLM 工具包）
- 生成时间：2026-09-08
- 运行方式：直接运行（`uv` + 本机 GPU；非 Docker）
- 硬件评估：**满足**（结论 + 依据见下）

## 硬件评估

| 项 | 本机 | 项目要求 |
| --- | --- | --- |
| GPU | RTX 4080 16GB（评估时已用约 3.5GB，剩余约 12.8GB） | 训练需 NVIDIA GPU；推理推荐 GPU |
| 内存 | 约 32GB（可用约 13GB） | 无硬性最低；推理够用 |
| 磁盘 | E: 约 418GB 可用 | 模型与依赖需数 GB～数十 GB |

依据：官方示例默认 `nvidia/parakeet-tdt-0.6b-v2`（约 0.6B），16GB 显存可舒适跑 ASR 推理。大模型微调 / SpeechLM2（如 Canary-Qwen-2.5B）或大批量训练可能吃紧，需另评估。

**Windows 注意**：`uv sync --extra cu12/cu13` 在 Windows 上会解析到 **CPU 版 PyTorch**（`sys_platform == 'linux'` 才装 CUDA wheel）。本机必须走「自带 CUDA 版 torch」路径，见下方。

## 环境准备

已选源（探测结果）：

- PyPI：清华镜像更快 → `https://pypi.tuna.tsinghua.edu.cn/simple`
- PyTorch wheel：直连 `download.pytorch.org` 可达
- Hugging Face：直连超时 → `HF_ENDPOINT=https://hf-mirror.com`
- 样例音频：AWS `dldata-public` 返回 403，改用 `hf-mirror.com/datasets/Narsil/asr_dummy`
- ffmpeg：`E:\Programs\ffmpeg-master-latest-win64-gpl\bin`（已在 `start.ps1` 注入 PATH）
- `nvcc`：未在 PATH（本机可不装 Toolkit；用 PyTorch 自带 CUDA 运行时即可）
- 加载 `.nemo`：`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`（仅信任来源）

```powershell
cd E:\AI\local-voice\Speech

# 1) 虚拟环境（Python >= 3.12）
uv venv --python 3.12

# 2) CUDA 版 PyTorch（Windows：勿用 index 裸装，会落到 +cpu；必须装带 +cu 的 wheel）
#    本机实测：2.9.1+cu128 / cp312 / win_amd64，驱动 CUDA 13.x 可用
$env:UV_HTTP_TIMEOUT = '300'
uv pip install `
  "https://download.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp312-cp312-win_amd64.whl" `
  "https://download.pytorch.org/whl/cu128/torchaudio-2.9.1%2Bcu128-cp312-cp312-win_amd64.whl"

# 3) 可编辑安装 NeMo Speech（先装 ASR；需要 TTS 再加 tts）
$env:UV_INDEX_URL = 'https://pypi.tuna.tsinghua.edu.cn/simple'
uv pip install -e ".[asr]"   # 已满足 torch>=2.7 时不会覆盖成 CPU 版

# 4) 模型下载走镜像
$env:HF_ENDPOINT = 'https://hf-mirror.com'
```

可选：`uv pip install -e ".[asr,tts]"` 以启用 Magpie TTS 演示。

本机当前栈：Python 3.12.13、`torch 2.9.1+cu128`、`nemo-toolkit` editable、`RTX 4080`。

## 启动

- 推荐：`pwsh -NoProfile -File .\start.ps1`
- 说明：启动后按提示选择（可单开）；默认 [1] 验证安装；不要默认全开
- 等价手动命令：

### verify（验证安装）

```powershell
$env:HF_ENDPOINT = 'https://hf-mirror.com'
# 勿用 uv run（会按 uv.lock 同步成 Windows CPU torch）
.\.venv\Scripts\python.exe -c "import torch; import nemo.collections.asr as nemo_asr; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print('NeMo Speech ASR OK')"
```

### asr-demo（Parakeet 推理）

```powershell
$env:HF_ENDPOINT = 'https://hf-mirror.com'
.\.venv\Scripts\python.exe scripts_local\asr_demo.py
```

（`scripts_local\asr_demo.py` 由启动脚本按需生成：下载短样例音频 → 加载 `nvidia/parakeet-tdt-0.6b-v2` → 打印转写。）

## 验证

- verify：打印 `cuda True`、`NVIDIA GeForce RTX 4080`、`NeMo Speech ASR OK`，退出码 0
- asr-demo：成功加载模型并对样例 wav 输出英文转写文本，无 CUDA OOM

## 备注

- 端口：本项目无常驻 HTTP 服务；菜单项均为一次性 CLI
- 不要在本机对 Windows 使用 `uv sync --locked --extra cu13` 或裸 `uv pip install torch --index-url .../cu12x`（会落到 `+cpu`）
- 不要用 `uv run` 启动本机 demo：会按 `uv.lock` 同步并覆盖成 CPU torch；请用 `.venv\Scripts\python.exe` 或 `start.ps1`
- Docker / NGC 镜像 `nvcr.io/nvidia/nemo-speech:26.07.00` 以 Linux 为主；本机优先直接运行
- 大模型（SpeechLM2 等）或量化方案需另开评估，勿默认同开多个占卡任务
- `docs/RUN.md` 与 `start.ps1` 为本机运行材料，默认不提交

## 审计备注（2026-09-09）
- 已修复 NeMo editable 映射 `E:\AI\local-voice\Speech` → `E:\AI\local-voice\Speech`。否则 `import nemo` 会指向不存在的旧路径。
