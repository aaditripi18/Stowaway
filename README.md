# Stowaway

**A model file that every scanner calls clean can still carry a payload.**

Model scanners look for *code* hidden in pickle data, and converting to
`.safetensors` removes code entirely, so safetensors files are trusted by default.
But a payload does not need to be code: it can sit in the low bits of the weights
(EvilModel), in bytes appended after the last tensor, or in alignment padding. It
survives format conversion because it is just numbers.

Stowaway looks for what the code scanners do not, and measures the limit: the
largest payload that no single-file check can see, and what hiding it costs the model.

> Status: week 4 of 8. All five checks work and are tested. Experiments so far:
> two ImageNet models end to end (E0-E2), and single-file checks on samples of 70
> public Hugging Face models (E0b, E1b, E1c). Preliminary findings are in
> [docs/FINDINGS.md](docs/FINDINGS.md), and every number, with the commit that
> produced it, in [docs/RESULTS.md](docs/RESULTS.md) (generated, never edited by
> hand). See [PLAN.md](PLAN.md).
>
> Early results: a 4 MB random payload in ResNet-18's low 8 bits leaves accuracy
> unchanged and passes every single-file check. Across 39 popular float32 models,
> the file alone exposes such a payload in 16 and says nothing about 23.

## Checks

| ID | Check | Status |
|---|---|---|
| S1 | Pickle imports, found by a restricted unpickler and an opcode walk; nothing is ever executed | done |
| S2 | Byte coverage: every byte belongs to a header, a tensor or zero padding | done |
| S3 | Precision history: low bits that should be zero, given how the model was trained | done |
| S4 | Bit-plane statistics per tensor (a biased plane or a non-uniform low byte) | done |
| S5 | Bit-level diff against the claimed base model (`--base`) | done |

A random payload in the low bits of a genuinely fp32-trained model is invisible to
single-file statistics (S3/S4) by design — that limit is what experiment E1 will
measure. When the claimed base model is available, S5 catches it regardless of size.

Formats: `.safetensors`, PyTorch zip (`.pt`/`.pth`/`.bin`), legacy PyTorch, plain
pickle (protocols 0–5) and GGUF. Formats are detected from content, never from the
extension.

## Run it

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"     # Windows; use .venv/bin/python elsewhere
.venv/Scripts/python -m pytest
.venv/Scripts/python -m stowaway scan model.safetensors
.venv/Scripts/python -m stowaway scan model.pt --json
```

Exit status: 0 clean, 1 suspicious, 2 dangerous.

## Guarantees, and how they are tested

- **The scanner never executes a pickle.** `find_class` only returns stubs defined
  in `stowaway/pickle_scan.py`. Tests pickle a payload that would create a file if
  run, scan it in every pickle protocol and both torch formats, and assert the file
  never appears.
- **Parsers agree with the reference implementations.** Every fixture is written by
  `torch`, `safetensors` or `gguf`, and every tensor's bytes, read through
  Stowaway's offsets, must equal the reference library's.

## Experiments

```bash
.venv/Scripts/python -m pip install -e ".[dev,experiments]"
export TORCH_HOME=$PWD/models/torch PYTHONPATH=.
.venv/Scripts/python scripts/run_e0.py                     # false positives, 9 clean models
.venv/Scripts/python scripts/run_e1.py --model resnet18    # detection vs payload size
.venv/Scripts/python scripts/run_e2.py --model resnet18    # accuracy cost (needs Imagenette in data/)
.venv/Scripts/python scripts/report.py                     # regenerate docs/RESULTS.md
```

The Hugging Face survey samples tensors with HTTP range requests (no full downloads):

```bash
.venv/Scripts/python scripts/select_hf_models.py   # only to rebuild the frozen list
.venv/Scripts/python scripts/run_hf_survey.py      # E0b + E1b, ~10 minutes
.venv/Scripts/python scripts/run_s3_widened.py     # E1c, from the survey cache
```

Nothing is trained: models are downloaded pretrained and only run forward. Every
run appends to `results/ledger.jsonl` with the git commit that produced it.
