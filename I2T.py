#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
I2T.py

Batch *residual* semantic description for 360-degree panoramic images (ERP)
using an OpenAI-compatible Chat Completions API with vision (e.g. laozhang.ai).

- .list 文件每行:  <orig_pano_path> <recon_pano_path>
- 原图:   high-quality ERP panorama
- 重建图: extremely low bit-rate reconstruction
- 输出: JSON, key = 原图绝对路径, value = {
      "semantic_residual": [...],
      "texture_residual": [...]
  }

用法示例:

  export OPENAI_API_KEY=sk-...
  python I2T.py \
      --list datalists/train_pairs.list \
      --root . \
      --out pano_semantic_residual.json \
      --model gpt-4o-mini \
      --max-workers 2 \
      --rate-limit 0.5 \
      --base-url https://api.laozhang.ai/v1
"""

import os
import sys
import time
import json
import base64
import argparse
import mimetypes
import threading
from typing import Iterable, List, Dict, Tuple

from openai import OpenAI


# =====================================================================
# Prompt: 提取“丢失的信息”，分 semantic / texture 两部分
# =====================================================================

DEFAULT_PANO_PROMPT = """
You are given TWO 360-degree equirectangular (ERP) panoramas of the same scene.

Image A = original high-quality panorama.
Image B = heavily compressed/reconstructed panorama at extremely low bit-rate.

Your task is to identify visual information that is clearly present in A
but missing, heavily blurred, distorted, or simplified in B.
This is the "residual information" that should be restored during decoding.

Important:

1. Output MUST be divided into two sections:
   [semantic_residual]  — objects, shapes, scene elements, and structures
                           that are present in A but no longer recognizable
                           or clearly visible in B.
   [texture_residual]   — fine textures, materials, edges, and high-frequency
                           details that are present in A but lost or strongly
                           degraded in B.

2. For each section, output a comma-separated list of 8–20 short phrases.
   Each phrase should directly name the objects or details that should be
   present, for example:
     "distant building windows", "small street signs", "tree branches",
     "human faces", "thin railings", "floor tile patterns",
     "sand grain texture", "grass leaf texture", "wood grain", "sky color gradient".
   Phrases should be concise (2–6 words).

3. Do NOT include words like "lost", "missing", "blurred", "degraded" in the phrases.
   Just describe what objects, details, or textures should be present and are
   no longer visible or clear in the reconstructed image B.

4. Do NOT output full sentences, JSON, or explanations.
   Only output the two sections with comma-separated phrases.

Output format:

[semantic_residual]
phrase1, phrase2, phrase3, ...

[texture_residual]
phrase1, phrase2, phrase3, ...
""".strip()


# =====================================================================
# 基础工具
# =====================================================================

def read_list_file(list_path: str) -> List[Tuple[str, str]]:
    """读取 <orig> <recon> 对"""
    pairs = []
    with open(list_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) != 2:
                raise ValueError(f"Line {idx} must have 2 paths: {s}")
            pairs.append((parts[0], parts[1]))
    return pairs


def image_path_to_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def build_user_content(prior_prompt: str, data_url_orig: str, data_url_recon: str):
    instr = (
        "You are an expert in 360-degree panoramic images and semantic compression. "
        "Carefully compare the original (Image A) and the compressed reconstruction (Image B).\n\n"
    )
    if prior_prompt:
        instr += prior_prompt

    return [
        {"type": "text", "text": instr},
        {
            "type": "image_url",
            "image_url": {"url": data_url_orig, "detail": "high"},
        },
        {
            "type": "image_url",
            "image_url": {"url": data_url_recon, "detail": "high"},
        },
    ]


def call_model(
    client: OpenAI,
    model: str,
    data_url_orig: str,
    data_url_recon: str,
    prior_prompt: str,
    timeout: float = 120.0,
) -> str:
    """调用带 vision 的 chat.completions，返回原始文本"""
    user_content = build_user_content(prior_prompt, data_url_orig, data_url_recon)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert assistant for 360-degree panoramas and semantic compression. "
                    "You only output semantic and texture residuals."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        timeout=timeout,
    )

    text = resp.choices[0].message.content or ""
    return text.strip()


def parse_residual_text(raw: str):
    """
    从模型输出里解析出两个 list：

    [semantic_residual]
    a, b, c

    [texture_residual]
    x, y, z
    """
    semantic, texture = [], []
    current = None

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue

        low = line.lower()
        if low.startswith("[semantic_residual]"):
            current = "sem"
            continue
        if low.startswith("[texture_residual]"):
            current = "tex"
            continue

        tokens = [t.strip() for t in line.split(",") if t.strip()]
        if current == "sem":
            semantic.extend(tokens)
        elif current == "tex":
            texture.extend(tokens)

    return semantic, texture


# =====================================================================
# 多线程 worker
# =====================================================================

def worker_loop(
    client: OpenAI,
    model: str,
    prior_prompt: str,
    img_queue: Iterable[Tuple[str, str]],
    results_dict: Dict[str, Dict[str, List[str]]],
    lock: threading.Lock,
    rate_limit_sec: float,
    retries: int,
    timeout: float,
):
    img_queue = list(img_queue)
    total = len(img_queue)

    for idx, (orig_path, recon_path) in enumerate(img_queue, start=1):
        abs_orig = os.path.abspath(orig_path)
        abs_recon = os.path.abspath(recon_path)

        print(f"[worker] ({idx}/{total}) {abs_orig}")

        data_url_orig = image_path_to_data_url(abs_orig)
        data_url_recon = image_path_to_data_url(abs_recon)

        delay = 1.0
        text = ""
        for attempt in range(1, retries + 1):
            print(f"[worker] ({idx}/{total}) call API, attempt {attempt}/{retries}")
            try:
                text = call_model(
                    client,
                    model,
                    data_url_orig=data_url_orig,
                    data_url_recon=data_url_recon,
                    prior_prompt=prior_prompt,
                    timeout=timeout,
                )
                break
            except Exception as e:
                print(f"[worker] API error ({attempt}/{retries}): {repr(e)}", file=sys.stderr)
                if attempt < retries:
                    time.sleep(delay)
                    delay = min(delay * 2.0, 30.0)

        if not text:
            print(f"[worker] ({idx}/{total}) empty response, skip.", file=sys.stderr)
            continue

        sem_list, tex_list = parse_residual_text(text)

        with lock:
            results_dict[abs_orig] = {
                "semantic_residual": sem_list,
                "texture_residual": tex_list,
            }

        if rate_limit_sec > 0:
            time.sleep(rate_limit_sec)


def chunkify(lst: List, n: int) -> List[List]:
    n = max(1, n)
    k, m = divmod(len(lst), n)
    out, start = [], 0
    for i in range(n):
        end = start + k + (1 if i < m else 0)
        out.append(lst[start:end])
        start = end
    return out


# =====================================================================
# Main
# =====================================================================

def main():
    p = argparse.ArgumentParser(
        description="Batch residual semantic description for 360° ERP images via Chat Completions API"
    )
    p.add_argument("--list", required=True, help=".list file, each line: <orig> <recon>")
    p.add_argument("--root", default=".", help="Root for relative paths (default: .)")
    p.add_argument("--out", default="pano_semantic_residual.json", help="Output JSON")
    p.add_argument("--model", default="gpt-4o-mini", help="Vision model name")
    p.add_argument("--prior", default="", help="Custom prior prompt (optional)")
    p.add_argument("--max-workers", type=int, default=1, help="Threads")
    p.add_argument("--rate-limit", type=float, default=0.0, help="Sleep seconds per request")
    p.add_argument("--retries", type=int, default=3, help="Retries per pair")
    p.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout")
    p.add_argument("--append", action="store_true", help="Append to existing JSON, skip existing keys")
    p.add_argument("--base-url", default="https://api.laozhang.ai/v1", help="OpenAI-compatible base URL")
    p.add_argument("--api-key", default=None, help="API key (default: use OPENAI_API_KEY env)")
    args = p.parse_args()

    list_path = os.path.abspath(args.list)
    out_path = os.path.abspath(args.out)
    root_abs = os.path.abspath(args.root)

    # 读取 pairs
    raw_pairs = read_list_file(list_path)
    img_pairs = []
    for orig_rel, recon_rel in raw_pairs:
        orig_path = orig_rel if os.path.isabs(orig_rel) else os.path.join(root_abs, orig_rel)
        recon_path = recon_rel if os.path.isabs(recon_rel) else os.path.join(root_abs, recon_rel)
        img_pairs.append((os.path.normpath(orig_path), os.path.normpath(recon_path)))

    if not img_pairs:
        print("No image pairs found.", file=sys.stderr)
        return 1

    # 结果字典
    results: Dict[str, Dict[str, List[str]]] = {}
    if args.append and os.path.exists(out_path):
        print(f"[INFO] Append mode: load {out_path}")
        with open(out_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        exist = set(results.keys())
        img_pairs = [(o, r) for (o, r) in img_pairs if os.path.abspath(o) not in exist]
        print(f"[INFO] Skip {len(exist)} existing, process {len(img_pairs)} new pairs")

    # prompt
    prior_prompt = args.prior.strip() if args.prior.strip() else DEFAULT_PANO_PROMPT

    # client
    if args.api_key:
        client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    else:
        client = OpenAI(base_url=args.base_url)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # 多线程
    lock = threading.Lock()
    n_workers = max(1, int(args.max_workers))
    chunks = chunkify(img_pairs, n_workers)

    threads = []
    for i, chunk in enumerate(chunks, start=1):
        if not chunk:
            continue
        t = threading.Thread(
            target=worker_loop,
            args=(
                client,
                args.model,
                prior_prompt,
                chunk,
                results,
                lock,
                args.rate_limit,
                args.retries,
                args.timeout,
            ),
            daemon=True,
            name=f"worker-{i}",
        )
        t.start()
        threads.append(t)
        print(f"Started worker-{i} with {len(chunk)} pairs.")

    for t in threads:
        t.join()

    # 写出 JSON
    print(f"\n[INFO] Done. Writing {len(results)} entries to {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("[INFO] Finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
