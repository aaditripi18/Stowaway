# Using Stowaway

Three ways in, all producing the same report: the command line, the Python
function behind it, and the browser scanner (`web/`, byte-identical reports).

## Install

```
pip install "git+https://github.com/aaditripi18/Stowaway"          # the scanner: needs only numpy
pip install "stowaway-scan[dev] @ git+https://github.com/aaditripi18/Stowaway"   # + test dependencies

docker build -t stowaway https://github.com/aaditripi18/Stowaway.git
docker run --rm -v "$PWD:/data" stowaway scan /data/model.safetensors
```

Python 3.10 to 3.13; tested on Linux, Windows and macOS. The scanner never
imports torch and never unpickles anything with real classes.

## Command line

```
stowaway scan FILE|hf:org/name[@revision] [...] [--base FILE|hf:org/name[@revision]|declared] [--json]
```

| Exit status | Meaning |
|---|---|
| 0 | clean: no finding at medium severity or above |
| 1 | suspicious: at least one medium finding |
| 2 | dangerous: at least one high or critical finding |
| 3 | could not run (unreadable file, Hub error, `--base declared` without a declared base) |

With several files the status is the worst of them. `--json` prints the report
(a list for several files). `hf:` targets and bases are resolved to a commit and
checked against the Hub's SHA-256 before use; `HF_TOKEN` is sent for gated
repositories; downloads are cached in `STOWAWAY_CACHE` (default
`~/.cache/stowaway/hub`).

## Python

```python
from stowaway.report import scan

report = scan("model.safetensors")                       # single-file checks
report = scan("finetune.safetensors", base="base.safetensors")   # adds S5
report = scan("shard-1.safetensors", base=["base-1.safetensors", "base-2.safetensors"])  # sharded base

from stowaway import hub
res = hub.resolve("hf:org/name@main")    # commit, files, sizes, SHA-256, declared base_model
paths = hub.download(res)                # verified local paths
```

`scan` returns a plain dict (the report below) and reads one tensor at a time,
so memory stays near the size of the largest tensor whatever the file size.
These two entry points are the supported API; everything else in the package is
internal and may change between versions.

## The report

```json
{
  "report_version": 1,
  "file": "model.safetensors",
  "sha256": "…",
  "size": 17739144,
  "format": "safetensors",
  "verdict": "clean",
  "tensors": {"count": 46, "bytes": 17713664, "dtypes": {"F32": 46}},
  "findings": [
    {"check": "S2", "severity": "high", "code": "trailing-data",
     "message": "4096 bytes after the end of the model, looks like Windows executable (MZ)",
     "detail": {"start": 17739144, "end": 17743240, "bytes": 4096, "magic": "Windows executable (MZ)", "…": "…"}}
  ]
}
```

`format` is one of `safetensors`, `pytorch-zip`, `torchscript`, `pytorch-legacy`,
`pickle`, `gguf`, `unknown`. Findings are sorted by severity (worst first), then
check, code and message. Severities: `info`, `low`, `medium`, `high`, `critical`;
`verdict` is `dangerous` if any finding is high or critical, `suspicious` if any
is medium, else `clean`. The CLI adds `source` and `base` (Hub commit and files)
to reports of `hf:` targets.

## Finding codes

| Check | Code | Severity | What it means |
|---|---|---|---|
| S1 | `dangerous-import` | critical | a pickle imports a known code-execution gadget (`os.system`, `builtins.exec`, `subprocess.*`, …) |
| S1 | `unknown-import` | medium | a pickle imports something outside what checkpoints need (it is also not a known gadget) |
| S1 | `malformed-pickle` | medium | the pickle stream does not parse cleanly |
| S2 | `trailing-data` | high / medium / low | bytes after the end of the model (high if non-zero and >= 1 KB or a known file signature) |
| S2 | `leading-data` | high / medium / low | bytes before the model starts |
| S2 | `unaccounted-bytes` | high / medium / low | bytes between parts that nothing in the format points to |
| S2 | `unreferenced-storage` | high / medium / low | part of a PyTorch storage that no tensor uses |
| S2 | `dirty-padding` | high | non-zero bytes in alignment padding |
| S2 | `unexpected-zip-entry` | medium | a file in a `.pt` archive that torch does not use |
| S2 | `zip-comment` | medium | a comment attached to a `.pt` archive |
| S3 | `precision-origin` | info | a float32 file whose weights carry only fp16/bf16 precision |
| S3 | `precision-anomaly` | high | in such a model, a tensor using low bits the rest leaves zero |
| S4 | `lowbit-signature` | info | the model's low bits carry a signature tensors are read against |
| S4 | `lowbit-anomaly` | medium | a tensor's low bits are nearer a fair coin than its role peers' (a hint from one file, not proof: see FINDINGS) |
| S5 | `low-bit-only-change` | high | a tensor changed from the base only in the way a low-bit payload changes it |
| S5 | `aligned-names` | info | tensors were paired with the base after renaming (wrapper prefix, `gamma`/`beta`) |
| S5 | `no-shared-tensors` | info | the base shares no tensor with the model |
| parse | `unknown-format` | medium | not a recognised model format |
| parse | `truncated`, `bad-header-length`, `bad-header-json`, `bad-tensor-entry`, `duplicate-keys`, `size-mismatch`, `tensor-out-of-bounds`, `overlapping-tensors` | high | the safetensors header is malformed or inconsistent |
| parse | `bad-gguf`, `bad-block` | high | the GGUF header is unreadable, or a tensor's size does not fit its block type |
| parse | `bad-alignment`, `unaligned-tensor` | medium | GGUF alignment is invalid, or a tensor does not respect it |
| parse | `gguf-version` | low | a GGUF version other than 2 or 3 |
| parse | `bad-zip`, `bad-local-header` | high | the `.pt` archive is unreadable |
| parse | `bad-storage` | high | a legacy-format storage cannot be located |
| parse | `no-data-pkl`, `structure-unreadable`, `missing-storage`, `bad-legacy` | medium | the checkpoint's structure cannot be rebuilt as torch would |
| parse | `compressed-entry` | low | a `.pt` entry is compressed (torch never does this) |
| parse | `unknown-dtype` | info / low | a tensor dtype the weight checks do not read |

S4 and S5 also report `detail` fields with the numbers they used (bit, z or
share, the reference, the threshold), so every verdict can be re-derived.
