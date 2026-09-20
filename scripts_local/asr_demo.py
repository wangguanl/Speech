"""One-shot ASR demo for local NeMo Speech (wanggang-run-oss)."""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

# AWS sample often returns 403 from some networks; use HF mirror dummy clip.
SAMPLE_URL = "https://hf-mirror.com/datasets/Narsil/asr_dummy/resolve/main/1.flac"
SAMPLE_PATH = Path(__file__).resolve().parent / "asr_dummy_1.flac"
MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"


def main() -> None:
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    if not SAMPLE_PATH.is_file():
        print(f"Downloading sample: {SAMPLE_URL}")
        req = urllib.request.Request(SAMPLE_URL, headers={"User-Agent": "nemo-speech-local-demo"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(SAMPLE_PATH, "wb") as out:
            out.write(resp.read())

    import torch
    import nemo.collections.asr as nemo_asr

    print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    print(f"Loading {MODEL_NAME} ...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
    transcript = asr_model.transcribe([str(SAMPLE_PATH)])[0].text
    print("transcript:", transcript)


if __name__ == "__main__":
    main()
