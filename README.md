# Stowaway

**Live site and in-browser scanner: https://stowaway-jade.vercel.app**

A model file that every scanner calls clean can still carry a payload. Model
scanners look for *code* hidden in pickle data, and converting to `.safetensors`
removes code entirely, so safetensors files are trusted by default. But a payload
does not need to be code: it can sit in the low bits of the weights, in bytes
appended after the last tensor, or in alignment padding, and it survives format
conversion because it is just numbers.

Stowaway looks where the code scanners do not, in `.safetensors`, PyTorch and
GGUF files, and measures the limit: how much can hide before a single-file check
notices, what hiding it costs the model, and what only a comparison with the base
model can catch.

> Status: week 5 of 8, preliminary. Two vision models end to end, single-file
> checks on samples of 120 public Hugging Face models (70 dev, 50 held out), and
> S5 calibrated on 20 real fine-tune/base pairs. One payload type (random bytes).

## Results

Every number below is generated from `results/ledger.jsonl` into
[docs/RESULTS.md](docs/RESULTS.md); the narrative is in
[docs/FINDINGS.md](docs/FINDINGS.md). The hypotheses were written in
[PLAN.md](PLAN.md) before any experiment ran.

**Across 39 popular float32 models, the file alone exposes a payload in 16 and
says nothing about 23.** Random bytes in the low k = 1, 4 or 8 bits of one sampled
tensor, on the 40 most-downloaded non-gated float32 safetensors models:

| What the weights look like | Models | Payload flagged from the file alone |
|---|--:|---|
| float32 file carrying only fp16 or bf16 precision | 14 | 14 / 14 at every k |
| float32 with a low-bit signature | 2 | 2 / 2 at every k |
| float32 with fair-coin low bits | 23 | **0 / 23** at every k |

**Whether one file is enough depends on the model.** Random payload in the low 8
bits, 1 KB to 1 MB, with top-1 accuracy on 500 Imagenette images:

| Model | Single-file checks (S3/S4) | Base-model diff (S5) | Top-1 accuracy, clean → with payload |
|---|---|---|---|
| ResNet-18 (fair-coin low bits) | **missed at every size** | caught at every size | 66.8% → 66.8%, up to 4 MB |
| MobileNetV3-Small (low-bit signature) | caught at every size | caught at every size | 66.2% → 66.2%, up to 1 MB |

| | Hypothesis | Verdict |
|---|---|---|
| H1 | S1 matches picklescan on code payloads, and every weight-payload model passes picklescan and a safetensors conversion | **not yet tested** (E5) |
| H2 | In a model stored at higher precision than it was trained in, a payload ≥ 1 KB is caught by S3 with no false positives | **supported**: 19 / 19 bf16 models, 0 false positives, 1 KB caught in all |
| H3 | In a genuinely float32-trained model, an encrypted low-bit payload cannot be seen from one file | **model-dependent**: true for 23 of 39 surveyed models and ResNet-18, false for the 2 with a signature |
| H4 | A diff against the claimed base model catches a payload in any tensor the fine-tune did not train | **supported, with limits**: 10 / 10 held-out pairs at k = 8, 16, 20 and 21; 0 / 10 inside a tensor the fine-tune did train |

What the numbers say:

- **A clean scan is not a clean model.** For most popular float32 models the
  low bits are fair coins, and a payload thin enough moves no statistic far
  enough. The check that does not depend on where the bytes go is the diff
  against a trusted base.
- **A third of "float32" uploads are really half precision.** 14 of 39 hold only
  fp16 or bf16 precision, leaving zero low bits exactly where a payload would go,
  and exactly where S3 sees it.
- **An attacker who knows the checks can evade them.** On MobileNetV3-Small,
  where the naive attack is caught, spreading the payload over one weight in 64
  keeps every statistic under the threshold at every size up to 1 MB (E3). The
  base diff catches it from 4 KB.
- **The strong check has a blind spot too.** A payload confined to tensors the
  distributor genuinely trained is camouflaged by the training, so S5 misses it.
- **Validated on unseen models.** Each detector was frozen, then run once on
  models committed before their weights were read: S5 raised 0 false positives
  on 15 held-out honest fine-tunes; S4's false-positive models on a fresh batch
  fell from 5 to 1 of 20 bf16 and 1 to 0 of 20 fp16.

## Use it

```
pip install -e .

python -m stowaway scan model.safetensors
python -m stowaway scan finetune.safetensors --base base.safetensors   # adds S5
python -m stowaway scan model.pt --json                                # full report
```

`scan` exits 0 when clean, 1 when suspicious, 2 when dangerous. Formats are
detected from content, never the extension: `.safetensors`, PyTorch zip
(`.pt`/`.pth`/`.bin`), legacy PyTorch, plain pickle (protocols 0–5) and GGUF.

| Check | What it reports |
|---|---|
| S1 pickle imports | unsafe imports in pickle data, found by a restricted unpickler and an opcode walk; nothing is ever executed |
| S2 byte coverage | every byte must belong to a header, a tensor or zero padding; trailing data, dirty padding and unreferenced storage are reported with offsets |
| S3 precision history | low bits that should be zero for the precision the model was trained in (e.g. bf16 weights saved as float32) |
| S4 bit-plane statistics | low bits pulled toward a fair coin, read against the tensor's role peers in the same model |
| S5 base-model diff | `--base`: tensors whose changes from the claimed base stay in the low bits, a cutoff calibrated on real fine-tunes |

**The scanner never executes a pickle.** `find_class` only returns stubs defined
in `stowaway/pickle_scan.py`. Tests pickle a payload that would create a file if
run, scan it in every pickle protocol and both torch formats, and assert the file
never appears. **Parsers agree with the reference implementations:** every
fixture is written by `torch`, `safetensors` or `gguf`, and every tensor's bytes,
read through Stowaway's offsets, must equal the reference library's.

## Website and in-browser scanner

`web/` is a static site: the project page (`index.html`) and a scanner
(`scan.html`) where anyone can drop a model file, optionally with its base model,
and get the same report as the command-line tool, without installing anything.

- **The scanner cannot send your file anywhere.** Both pages carry a
  Content-Security-Policy with `connect-src 'none'` and `default-src 'none'`, and
  `vercel.json` sends the same policy as an HTTP header. Fonts and styles are
  self-hosted, so nothing loads from another origin. Checked in Chromium:
  `fetch` and `sendBeacon` are refused. `web/test/no_egress.test.ts`
  checks the policy and that `web/src/` uses no network API.
- **Same answers as Python.** `web/src/` is a TypeScript port of every check and
  parser. On every fixture the two must produce byte-identical canonical JSON
  reports, enforced in both directions (`web/test/*.test.ts` against committed
  golden reports, `tests/test_web_parity.py` against fresh Python output).
- The browser reads everything except legacy (pre-1.6) torch files and
  compressed zip entries, which the command-line tool handles.

```
cd web
npm ci
npm test            # 40 tests
npm run build       # static site in web/dist
```

Deployed on Vercel from `web/` (`cd web && vercel deploy --prod`); any static host
works. Design from Stitch; the figures on the project page are copied from
[docs/RESULTS.md](docs/RESULTS.md) and checked by hand, not yet generated.

## Limitations

- **Preliminary:** two vision models end to end, one seed, random payloads only.
  Accuracy cost is measured only on those two models.
- **Single-file statistics cannot see** a small or thinly spread encrypted
  payload in a fair-coin model. That is a finding, not a bug to tune away.
- **The base diff needs the real base model**, and misses a payload inside a
  tensor the fine-tune genuinely trained. A 1 KB payload spread at 1:64 also
  falls under its 256-changed-elements-per-tensor floor.
- **S4 trades detections for fewer false alarms:** it flags only bit planes
  pushed *toward* 0.5, so constant, zero or text payloads are left to S5.
- **The survey tests 64 KB samples, not whole files**, and is weighted to the
  most downloaded models, not a random sample.
- Open items are listed in [docs/FINDINGS.md](docs/FINDINGS.md).

## Reproduce

Everything runs on a laptop CPU; nothing is trained. Models are downloaded
pretrained and only run forward, and the Hugging Face survey samples tensors
with HTTP range requests instead of downloading whole files.

```
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,experiments]"   # Windows; .venv/bin/python elsewhere
.venv/Scripts/python -m pytest                                 # 106 tests

export TORCH_HOME=$PWD/models/torch PYTHONPATH=.
.venv/Scripts/python scripts/run_e0.py                      # false positives, 9 clean models
.venv/Scripts/python scripts/run_e1.py --model resnet18     # detection vs payload size
.venv/Scripts/python scripts/run_e2.py --model resnet18     # accuracy cost (needs Imagenette in data/)
.venv/Scripts/python scripts/run_e3.py --model mobilenet_v3_small   # adaptive placements
.venv/Scripts/python scripts/run_hf_survey.py --list hf_models.json          # E0b + E1b, dev
.venv/Scripts/python scripts/run_hf_survey.py --list hf_models_batch3.json   # held-out batch
.venv/Scripts/python scripts/compare_s4_rules.py --list hf_models_batch3.json   # E0c
.venv/Scripts/python scripts/run_s3_widened.py              # E1c, from the survey cache
.venv/Scripts/python scripts/run_e4.py --split dev          # S5 calibration
.venv/Scripts/python scripts/run_e4.py --split heldout      # S5 validation
.venv/Scripts/python scripts/report.py                      # docs/RESULTS.md
```

Every run appends a line to `results/ledger.jsonl` with the git commit that
produced it; `docs/RESULTS.md` is generated from it and never edited by hand. The
model lists (`results/hf_models*.json`, `results/ft_pairs_*.json`) were frozen
and committed before their weights were read, and each held-out batch was used
once.

## License

Code: MIT, see [LICENSE](LICENSE). The fonts bundled with the browser build keep
their own licenses (SIL OFL 1.1 and Apache 2.0); see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No model weights are
redistributed; test fixtures are small synthetic files.
