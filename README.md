# Stowaway

**Live site and in-browser scanner: https://stowaway-jade.vercel.app**

A model file that every scanner calls clean can still carry a payload. Model
scanners look for *code* hidden in pickle data, and converting to `.safetensors`
removes code entirely, so safetensors files are trusted by default. But a payload
does not need to be code: it can sit in the low bits of the weights, in bytes
appended after the last tensor, or in alignment padding, and it survives format
conversion because it is just numbers.

Stowaway looks where the code scanners do not, in `.safetensors`, PyTorch and
GGUF files, measures the limit of what any scanner can see, and then closes the
gap the other way: **where a payload cannot be detected, it removes it.**
`stowaway sanitize` rounds away the low bits of every float32 weight; on 10 of 10
models (nine image models and a text classifier) that left 100.00% of
predictions unchanged and 0 bytes of a planted 64 KB payload.

> Status: phases 1-5 done. Single-file checks on 260 public Hugging Face models
> (a top-1000 list, three held-out batches, a random sample of the top 10,000; 67
> scanned as whole files), S5 calibrated and validated on real fine-tune/base
> pairs, automatic reference discovery on the 23 models a file alone cannot
> judge, and the cost of sanitising on ten models. Phase 4 closed the three gaps
> whole files exposed, each with a rule chosen on seen models and run once on a
> fresh list: a windowed S4 (65 fresh whole files), S5 without its size floor
> (15 fresh pairs) and S5d (25 fresh pairs). Every hypothesis was written before
> its run; four of phase 3's eight failed and are reported as such, all six of
> phase 4's held. Phase 5 checks that a given base is close enough for S5 to mean
> anything (E15, 19 fresh pairs labelled by
> [Pedigree](https://github.com/aaditripi18/Pedigree)), and measured whether a
> rule can tell a never-trained tensor from a payload (E14: not safely), and drops
> whole-tensor S4 where the windowed test already looks (E17, 60 fresh whole files:
> every catch kept, clean models flagged 5 -> 2).

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
| H1 | S1 matches picklescan on code payloads, and every weight-payload model passes picklescan and a safetensors conversion | **supported** (E5): same 8 / 8 code payloads as picklescan; 4 / 4 weight-payload files pass picklescan; the payload survives conversion bit for bit in 2 / 2 |
| H2 | In a model stored at higher precision than it was trained in, a payload ≥ 1 KB is caught by S3 with no false positives | **supported**: 19 / 19 bf16 models, 0 false positives, 1 KB caught in all |
| H3 | In a genuinely float32-trained model, an encrypted low-bit payload cannot be seen from one file | **model-dependent**: true for 23 of 39 surveyed models and ResNet-18, false for the 2 with a signature |
| H4 | A diff against the claimed base model catches a payload in any tensor the fine-tune did not train | **supported, with limits** (whole files, E4b): 9 / 9 fresh pairs at k = 8, 16, 20 and 21, with 0 of 20 honest fine-tunes flagged; inside a tensor the fine-tune did train, 18 / 20 at k = 21 but 1 / 20 at k = 8 |

**Phase 3: closing the blind spot** (E8-E10, [docs/RESULTS.md](docs/RESULTS.md)):

| Question | Result |
|---|---|
| Does clearing the low 8 bits of every weight cost anything? (E9) | **No: 100.00% of predictions unchanged on 10 / 10 models**, identical accuracy; a 64 KB payload planted there leaves **0 / 65,536 bytes** |
| ...and the low 16 bits? | Yes: 92.4-100% agreement, under 99% on 8 of 10 models; 8 bits is the safe setting |
| How often can a reference model be found with no user input? (E8) | **18 of 20** held-out blind models: 5 through the declared `base_model`, 13 discovered by weight similarity (median cosine >= 0.9 among same-architecture Hub models) |
| Does the diff against that reference catch a payload? | In a tensor the two share: **12 / 13** (the miss is 84 bytes, under S5's old floor, removed in E12). In a tensor training changed: **0 / 14**, which is why sanitising exists |
| Is the blind spot an artefact of the most-downloaded models? (E10 B) | No, it is larger: **29 of 39 (74%)** float32 models of a random sample of the top 10,000 are fair-coin, against 59% in the top 1,000; 0 false positives on all 40 |
| Do 32 KB samples tell the truth about whole files? (E10 A) | Mostly: the model's group agrees on **67 / 67**, payload verdicts on 163 / 173. Whole files also show what samples hid: **5 of 67 clean models flag**, every inspected case a tensor whose precision history really differs (a never-trained tensor, or a component saved at full float32) |

| | Hypothesis (phase 3) | Verdict |
|---|---|---|
| H7 | At least half of the blind models declare a usable base | **not supported**: 5 of 23 |
| H7b | Declared plus discovered references cover at least half | **supported**: 18 of 20 held out |
| H8 | On those pairs S5 has no false positive and catches every payload in a shared tensor | **not supported**: 2 of 21 pairs flag a LayerNorm weight; 12 / 13 caught |
| H9 | Clearing 8 low bits keeps >= 99.9% agreement on every model | **supported**: 100.00% on 10 / 10 |
| H10 | Clearing 16 low bits keeps >= 99% on every model | **not supported**: under 99% on 8 of 10 |
| H11 | Samples and whole files agree on >= 95% of groups and verdicts | **not supported**: groups 67 / 67, verdicts 163 / 173 (94.2%) |
| H12 | The random sample's fair-coin share is within 15 points of the top 1,000's | **not supported**: 74% vs 59% (+15.4) |

Phase 4 fixed what H11 and E8 exposed, each on a fresh list:

| Question | Result |
|---|---|
| Does a windowed S4 restore whole-file sensitivity? (E11) | **Yes: whole file and sample agree on 163 / 168 planted cells** (160 with S3/S4 alone); in all 5 differences the whole file catches what the sample missed. 0 of 65 clean models gain a flag |
| Can S5 judge tensors with few changes? (E12) | **Yes: a thin payload (4 or 16 bytes per tensor) caught in 18 / 18 models**, 0 / 18 with the old 256-element floor; no honest pair gains a flag |
| Can S5 tell a LayerNorm shift from a payload? (E13) | **Yes: S5d** asks whether any changes carry above the payload's top bit. On seen pairs it removes all 56 honest flags in 4 pairs; on 25 fresh pairs it keeps **48 / 48** catches in base-matching tensors and the thin payload in **30 / 30** models |

| | Hypothesis (phase 4) | Verdict |
|---|---|---|
| H13 | Without the floor, S5 flags no honest pair the floor does not | **supported**: 0 added of 15 |
| H14 | Without the floor, S5 catches the thin payload in >= 17 of 18 models | **supported**: 18 / 18 at 16 and at 4 bytes per tensor |
| H15 | With S4w, whole file and sample agree on >= 95% of planted cells | **supported**: 163 / 168 (97.0%) |
| H16 | S4w flags a clean tensor in at most 1 fresh model | **supported**: 0 of 65 |
| H17 | S5d flags an honest tensor in at most 1 of 25 fresh pairs | **supported**: 1 (a 1-ULP re-save, flagged by S5c too) |
| H18 | S5d keeps >= 95% of S5c's base-matching catches and catches the thin payload in >= 95% of models | **supported**: 48 / 48; 30 / 30 and 30 / 30 |

Phase 5 (E15): S5 now reports how close the base is (median cosine of the aligned
tensors) and warns below 0.9, because against a base that is not the model's parent
S5's silence means nothing. E8's name-and-shape check had accepted such bases.

| | Hypothesis (phase 5) | Verdict |
|---|---|---|
| H19 | The warning fires on every independent same-architecture base, all of which pass the name-and-shape check | **supported**: 9 / 9 |
| H20 | It fires on no confirmed parent S5 can use | **supported**: 0 of 4; it fires on 3 confirmed but distant bases (cosine 0.10-0.89), where its message is right |
| H23 | Skipping whole-tensor S4 where S4w looks (S4x) keeps >= 99% of catches | **supported**: 20 / 20 |
| H24 | S4x flags a clean tensor in fewer fresh models | **supported**: 5 -> 2 (left: two embedding tables, one 256-element bias) |

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
  keeps every statistic under the threshold at every size that fits at that
  density, up to 16 KB (E3). The
  base diff catches it from 4 KB.
- **The strong check has a blind spot too.** A payload confined to tensors the
  distributor genuinely trained is camouflaged by the training, so S5 misses it.
- **The base diff passed on samples and failed on whole files, then was fixed.**
  E4 validated S5 on sampled tensors (0 false positives on 15 pairs). Scanning
  whole files, the same rule flagged 8 of 12 dev and 16 of 20 fresh honest
  fine-tunes: tensors that barely train (attention key biases, LayerNorm) change
  only in their low bits. The adopted rule (S5c) asks whether about half the
  changes stop at one bit, as a payload's do: chosen on the 12 dev pairs, run once
  on 20 fresh ones, it flagged 0 and still caught every payload in a tensor
  identical to the base (E4b). Later lists found LayerNorm gains that S5c still
  flagged; S5d adds that a low-bit payload changes nothing above its top bit,
  while a training update carries into higher bits (E13).
- **What cannot be detected can be removed.** A payload in the low bits of a
  fair-coin model, or of a tensor a fine-tune trained, is indistinguishable from
  the noise it replaces, by any check of one file or of a diff. Rounding those
  bits away destroys it, and on every model measured the low 8 bits carry no
  information the predictions use (E9).
- **A model's parent can usually be found, but "found" means the closest
  relative.** Discovery returned a parent, a fine-tune of the model, or a mirror
  copy; all are fair references for tensors they share, none proves where the
  model came from. A declared base can also mislead: MiniLM-L4's declared base
  passes a name-and-shape check with a weight cosine of 0.07 (E8).
- **Scanning whole files costs false alarms that samples hide.** Never-trained
  tensors (T5's cross-attention bias, XLM-R's pooler, ESM2's unused position
  table) and CLIP-ViT-L's full-precision text tower are flagged correctly as
  different from the rest of their file. A payload would look the same.
- **Validated on unseen models.** Each detector was frozen, then run once on
  models committed before their weights were read: S5c 0 of 20 fresh fine-tunes
  flagged; S4's false-positive models on a fresh batch fell from 5 to 1 of 20
  bf16 and 1 to 0 of 20 fp16 (E0c), and on a fourth batch S4 flags 2 of 20
  float32, 1 of 20 bf16 and 0 of 20 fp16 models (E0d).

## Install

```
pip install "git+https://github.com/aaditripi18/Stowaway"     # needs only numpy
docker build -t stowaway https://github.com/aaditripi18/Stowaway.git
docker run --rm -v "$PWD:/data" stowaway scan /data/model.safetensors
```

Python 3.10–3.13 on Linux, Windows and macOS (CI). The supported Python API,
the report schema and every finding code are in [docs/API.md](docs/API.md);
changes are in [CHANGELOG.md](CHANGELOG.md). Distributed from GitHub only.

## How it relates to existing scanners

| | picklescan, ModelScan, fickling | Stowaway |
|---|---|---|
| Code in pickle data (S1) | yes: their core job | yes; flags the same 8/8 code payloads as picklescan in E5 |
| `.safetensors` and `.gguf` | not read (no pickle inside) | parsed, and every byte accounted for |
| Bytes outside the format (S2) | no | trailing data, padding, unused storage, extra zip entries |
| Data in the weights (S3, S4) | no | yes, with the limits measured: from one file it often cannot see a payload |
| Diff against the claimed base (S5) | no | yes, including straight from the Hub against the card's `base_model` |
| Runs in the browser, nothing uploaded | no | yes, with the same reports |

Stowaway is not a replacement for a pickle scanner's maintained denylist, and
nothing here proves a model safe: a clean report means these checks found
nothing, and FINDINGS says what each check cannot see.

## Use it

```
pip install -e .

python -m stowaway sanitize model.safetensors --out clean.safetensors  # remove any low-bit payload
python -m stowaway scan model.safetensors
python -m stowaway scan finetune.safetensors --base base.safetensors   # adds S5
python -m stowaway scan model.pt --json                                # full report

# straight from the Hugging Face Hub, against the base model its card declares
python -m stowaway scan hf:org/finetune --base declared
python -m stowaway scan hf:org/finetune@v1.2 --base hf:org/base@main
```

`sanitize` rounds every float32 weight to nearest at mantissa bit 8 (`--bits`),
keeps NaN and infinity, never rounds a finite weight to infinity, and copies every
other tensor, the header and the metadata byte for byte. `scan` exits 0 when
clean, 1 when suspicious, 2 when dangerous, 3 when it could not run. Formats are detected from content, never the extension: `.safetensors`,
PyTorch zip (`.pt`/`.pth`/`.bin`), legacy PyTorch, plain pickle (protocols 0–5)
and GGUF.

- **Any size.** Both the CLI and the browser hold one tensor (and its base
  counterpart) in memory at a time: a 3.19 GiB model scans with 0.375 GiB peak
  memory in Python and 0.17 GiB in the browser core, with the same report.
- **Hub models are pinned.** An `hf:` target or base is resolved to a commit,
  each file is checked against the SHA-256 the Hub publishes before it is used
  (a short or altered download is refetched, then refused), and the report
  records the commit. Sharded checkpoints are read as one model. `HF_TOKEN` is
  sent for gated repositories. Downloads are never loaded or unpickled.
- **Fine-tunes that rename the base still pair up.** S5 aligns tensor names
  across wrapper prefixes (`bert.x` saved as `encoder.x`) and old LayerNorm
  names (`gamma`/`beta`), and says so in the report when it did.

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

`web/` is a static site: the project page (`index.html`), a scanner
(`scan.html`) where anyone can drop a model file, optionally with its base model,
and get the same report as the command-line tool, and **a limits page
(`limits.html`)** where a reader plants random bytes in a model (a bundled
Apache-2.0 BERT or their own file), chooses size, depth, density and target
tensors, sweeps the grid, and downloads the planted copy to check with the CLI.
The planting code is a port of `stowaway.embed`, byte-identical in tests.

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
- **Any size.** The scan runs in a worker that reads the file a range at a
  time (`FileReaderSync` over `File.slice`), so memory is about one tensor, not
  the file: a 3.19 GiB model scans with 0.17 GiB peak memory and the same
  report as Python (`web/test/large_file.test.ts`, opt-in, with
  `scripts/make_large_fixture.py`). The worker starts from a `blob:` URL, so it
  inherits the page policy; `fetch`, `WebSocket` and `importScripts` are
  refused inside it.
- Every format the Python tool reads, including legacy (pre-1.6) torch files
  and deflate-compressed zip entries. A `data.pkl` compressed with bzip2 or
  LZMA (which torch never writes) is reported as unreadable in the browser.

```
cd web
npm ci
npm test            # 124 tests
npm run build       # static site in web/dist
```

Deployed on Vercel with `web/` as the project root: every push to `main` builds
and publishes it (`cd web && vercel deploy --prod` does the same by hand). Any
static host works. Design from Stitch. Every number on the project page is computed from
the ledger by `scripts/site_facts.py` (`--write` rewrites them); CI fails if any number on
the page differs from what the ledger gives.

## What it can and cannot see, and what to do about it

Each limit below was measured; where detection cannot work, the answer is
removal.

- **No check of one file can see a low-bit payload in a fair-coin model** (74%
  of float32 models in a random sample). Not a bug to tune away: such bits carry
  no structure to compare against. `sanitize` removes the payload instead, at no
  measured cost with 8 bits; 16 bits changes up to 7.6% of predictions.
- **A diff cannot see a payload inside a tensor training changed** (0 / 14 in
  E8): training randomises the same low bits. Again, `sanitize`.
- **Reference discovery finds a close relative, not provenance**, and found none
  for 2 of 20 models; it needs a same-architecture model on the Hub. A base you
  pass yourself is checked: below a median cosine of 0.9 the report says S5 can
  only see the tensors identical to it (E15).
- **Whole files flag 5 of 67 clean popular models**: tensors whose precision
  history genuinely differs from the rest of the file. A report says which tensor
  and why; it cannot say whether the cause is benign. E14 measured whether one
  could: a never-trained tensor's values look like a fresh random draw, but so do
  about one in 26 trained tensors, so no rule on values can excuse the first
  without hiding payloads in the second. `sanitize` settles it either way. Flags
  on large tensors S4w already tests are gone since E17 (S4x: fresh models with
  such a flag 5 -> 2, every catch kept); the rest are named in the report.
- **A change of one unit in the last place looks like a 1-bit payload.** One of
  25 fresh honest pairs re-saved a tensor so that 8,265 elements differ in bit 0
  only; at the bit level that is what a 1-bit payload is, and S5 flags it (E13).
  The report names the tensor and the bit, so a reviewer can see it is a 1-ULP
  change; `sanitize` clears it either way.
- **A tensor nudged by rounding looks like a tiny payload.** On a fresh pair
  (E15), S5 flagged 8 RMSNorm weights of Qwen2-0.5B-Instruct against its true
  parent: 1 to 3 of 896 elements each, one unit in the last place. A one-bit
  change is what a one-bit payload is; the report shows the count and the bit,
  and `sanitize` clears it.
- **Inside tensors training changed, S5d catches fewer payloads than S5c** (46
  against 58 of 96 cells, E13): those are the payloads E8 showed a diff cannot
  reliably see, and `sanitize` is the answer there.
- **Sanitising was measured on ten models with 1,000 images or 872 sentences
  each**; a model whose predictions depend on the lowest float32 bits would lose
  accuracy, so the agreement check in E9 is worth repeating on your own model.
- Every earlier gap and the experiment that closed it are listed in [docs/FINDINGS.md](docs/FINDINGS.md#every-earlier-gap-and-how-it-was-closed).

## Reproduce

Everything runs on a laptop CPU; nothing is trained. Models are downloaded
pretrained and only run forward, and the Hugging Face survey samples tensors
with HTTP range requests instead of downloading whole files.

```
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,experiments,compare]"   # Windows; .venv/bin/python elsewhere
.venv/Scripts/python -m pytest

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
.venv/Scripts/python scripts/run_e5.py                      # vs picklescan, ModelScan, fickling
.venv/Scripts/python scripts/run_e9.py                      # cost of sanitising, 9 vision + 1 text model
.venv/Scripts/python -m scripts.run_hf_survey --list hf_models_random.json   # E10 B, random sample
.venv/Scripts/python scripts/kaggle_e10.py run e8           # E8 on a Kaggle CPU session (pull e8 after)
.venv/Scripts/python scripts/kaggle_e10.py run e10a         # E10 A, whole files (pull e10a after)
.venv/Scripts/python scripts/report.py                      # docs/RESULTS.md
.venv/Scripts/python scripts/site_facts.py                  # page numbers == ledger (--write to update)
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
