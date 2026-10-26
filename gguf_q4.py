"""SmolVLM-256M as a 4-bit GGUF model on the llama.cpp CUDA backend (issue #1 of edge-ai-smol-vlm-256-optimization).

- Downloads the f16 language model and the f16 vision projector (mmproj) from ggml-org/SmolVLM-256M-Instruct-GGUF.
- Quantizes the language model with llama-quantize (q4_k_m by default, also q4_0, q8_0 and f16 as references).
  The vision projector stays f16 in every setup, so only the language model changes.
- Runs llama-mtmd-cli with all layers on the GPU (-ngl 99). The timings come from llama.cpp's own log, so they are
  the same numbers llama.cpp reports (image encode, prompt eval, decode).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

import numpy as np

from smolvlm_cuda_graphs import PROMPT

HF = "https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main"
F16_MODEL = "SmolVLM-256M-Instruct-f16.gguf"
MMPROJ = "mmproj-SmolVLM-256M-Instruct-f16.gguf"

RE_ENCODE = re.compile(r"mtmd batch encoding done in ([\d.]+) ms")
RE_PREFILL = re.compile(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens")
RE_DECODE = re.compile(r"(?<!prompt )eval time =\s*([\d.]+) ms /\s*(\d+) runs")


def download(name, dst):
    if dst.exists():
        return
    print(f"downloading {name} ({HF}/{name})")
    tmp = dst.with_suffix(".part")
    urllib.request.urlretrieve(f"{HF}/{name}", tmp)
    tmp.rename(dst)


def quantize(tool, src, quant, workdir):
    dst = workdir / f"SmolVLM-256M-Instruct-{quant.upper()}.gguf"
    if quant == "f16":
        return src
    if not dst.exists():
        subprocess.run([tool, str(src), str(dst), quant.upper()], check=True, capture_output=True)
    return dst


def run_once(cli, model, mmproj, image, prompt, n_predict, ngl):
    cmd = [cli, "-m", str(model), "--mmproj", str(mmproj), "--image", str(image), "-p", prompt,
           "-n", str(n_predict), "-ngl", str(ngl), "--temp", "0", "-v"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError((r.stderr.strip().splitlines() or ["llama-mtmd-cli failed"])[-1])
    log = r.stderr
    encode = sum(float(m) for m in RE_ENCODE.findall(log))
    pre = RE_PREFILL.findall(log)
    dec = RE_DECODE.findall(log)
    if not pre or not dec:
        raise RuntimeError("no timing lines in llama.cpp output (run with -v)")
    prefill_ms, _ = float(pre[-1][0]), int(pre[-1][1])
    decode_ms, tokens = float(dec[-1][0]), int(dec[-1][1])
    return {"encode_ms": encode, "prefill_ms": prefill_ms, "decode_ms": decode_ms, "decode_tokens": tokens,
            "answer": r.stdout.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--quants", default="f16,q8_0,q4_k_m", help="comma list: f16, q8_0, q4_k_m, q4_0")
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--runs", type=int, default=3, help="runs per setup (run 1 is a warm-up and is dropped)")
    ap.add_argument("--ngl", type=int, default=99, help="layers on the GPU (99 = all)")
    ap.add_argument("--workdir", default="gguf")
    ap.add_argument("--cli", default="llama-mtmd-cli", help="path to the llama.cpp multimodal CLI")
    ap.add_argument("--quantize", default="llama-quantize", help="path to llama-quantize")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    for tool in (args.cli, args.quantize):
        if shutil.which(tool) is None and not Path(tool).exists():
            raise SystemExit(f"{tool} not found: build llama.cpp (see the notebook) or pass --cli / --quantize")

    work = Path(args.workdir)
    work.mkdir(exist_ok=True)
    f16 = work / F16_MODEL
    mmproj = work / MMPROJ
    download(F16_MODEL, f16)
    download(MMPROJ, mmproj)

    Path(args.out).mkdir(exist_ok=True)
    rows, ref = [], None
    print(f"{'quant':<8} {'LM MB':>6} {'encode ms':>9} {'prefill ms':>10} {'tokens':>6} {'decode ms/tok':>13} "
          f"{'total ms':>9} {'same as f16':>11}  answer")
    with open(Path(args.out) / "gguf_q4.jsonl", "w") as f:
        for quant in args.quants.split(","):
            try:
                model = quantize(args.quantize, f16, quant, work)
                runs = [run_once(args.cli, model, mmproj, args.image, args.prompt, args.n_predict, args.ngl)
                        for _ in range(args.runs + 1)][1:]
            except Exception as e:
                print(f"{quant:<8} failed: {type(e).__name__}: {e}")
                continue
            mean = {k: float(np.mean([r[k] for r in runs])) for k in ("encode_ms", "prefill_ms", "decode_ms", "decode_tokens")}
            answer = runs[-1]["answer"]
            ref = ref or answer
            per_tok = mean["decode_ms"] / max(mean["decode_tokens"], 1)
            total = mean["encode_ms"] + mean["prefill_ms"] + mean["decode_ms"]
            row = {"quant": quant, "lm_mb": round(os.path.getsize(model) / 2**20, 1), **{k: round(v, 2) for k, v in mean.items()},
                   "decode_ms_per_token": round(per_tok, 2), "total_ms": round(total, 1),
                   "same_as_f16": answer == ref, "answer": answer}
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            print(f"{quant:<8} {row['lm_mb']:6.0f} {row['encode_ms']:9.1f} {row['prefill_ms']:10.1f} "
                  f"{row['decode_tokens']:6.1f} {per_tok:13.2f} {total:9.1f} {str(row['same_as_f16']):>11}  {answer[:50]}")
    if rows:
        best = min(rows, key=lambda r: r["total_ms"])
        print("SUMMARY", json.dumps({"best_quant": best["quant"], "total_ms": best["total_ms"],
                                    "rows": [{k: r[k] for k in ("quant", "total_ms", "decode_ms_per_token", "same_as_f16")} for r in rows]}))


if __name__ == "__main__":
    main()
