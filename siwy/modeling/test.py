from pathlib import Path

from loguru import logger

artifact_dir = "D:\SIWY_BJ\SIWY-25Z\\artifacts\cat-dog-2025-12-23-17-17-44-model-0-epoch-7-v0"
logger.info(f"Downloaded artifact to: {artifact_dir}")
# Pobierz wszystkie checkpointy *.pt z artifactu (np. epoki 1-7)
# ckpt_files = sorted([str(p) for p in Path(artifact_dir).rglob("*.pt") if any(f"epoch_{ep}" in str(p) for ep in range(1, 8))])
ckpt_files = list(Path(artifact_dir).glob("./*.pt"))
logger.info(f"ckpt_files: {ckpt_files}")
