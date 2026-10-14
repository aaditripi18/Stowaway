# Stowaway — plan

**Question:** when a model file is "clean" by every existing scanner, what can still
be hiding in it, and how much can be hidden before anyone could notice?

## The gap

Model scanners (picklescan, ModelScan, Hugging Face's own pickle scanning) look for
*code*: dangerous imports in pickle data. Converting to `.safetensors` removes that
risk entirely, so a safetensors file is treated as clean by definition.

But a payload does not have to be code. EvilModel (Wang et al., 2021) and follow-ups
hid working malware in the low-order bits of a network's weights with little loss of
accuracy. Those bits are just numbers: they survive conversion to safetensors, and no
code scanner looks at them. A loader that later extracts and runs them is small and
can ship separately.

Two more places nobody checks:

- **Bytes the format does not account for.** Data appended after the last tensor,
  junk in alignment padding, extra entries in a `.pt` zip, a zip comment. A parser
  that reads only what the header points to never sees them (Remnant's
  "video appended to a photo", for model files).
- **The claimed base model.** A fine-tune that says `base_model: X` should differ from
  X only in the layers it trained. Any bit-level change in a layer that should be
  frozen is unexplained.

## What Stowaway checks

| ID | Check | Needs | Catches |
|---|---|---|---|
| S1 | Pickle import analysis, without executing anything | `.pt`/`.bin`/`.ckpt` | code payloads (parity with picklescan) |
| S2 | Byte coverage: every byte must belong to a header, a tensor or expected padding | any format | appended data, dirty padding, extra zip entries |
| S3 | Precision history: do the low bits look like the precision the model was trained in? | float tensors | payloads in models trained in bf16/fp16 and stored in fp32 (low 16 bits should be zero) |
| S4 | Bit-plane statistics per tensor: uniformity, runs, correlation with the high bits | float tensors | payloads in genuinely fp32-trained models, if they are detectable at all |
| S5 | Differential: compare against the claimed base model, tensor by tensor, bit by bit | the base model | low-bit changes in layers that should be untouched |

S3 and S4 are the research part. S5 is the strongest check when a base model exists.

## Hypotheses (written before any experiment)

- **H1.** Stowaway's S1 matches picklescan on code payloads, and every weight-payload
  model passes picklescan and a safetensors conversion untouched.
- **H2.** In models stored at a higher precision than they were trained in, any
  weight payload of 1 KB or more is detected by S3 with no false positives on clean
  models.
- **H3.** In genuinely fp32-trained models, an *encrypted* payload in the lowest bits
  is not detectable by single-file statistics (S4) below some size N, and N is large
  enough to hold real malware with < 1% accuracy loss. If true, this is the headline:
  single-file scanning cannot certify a fp32 model, and only S5 can.
- **H4.** S5 detects any payload in a layer the fine-tune did not train, at any size.

H3 might come out either way. Both outcomes are findings; the plan does not assume
detection works.

## Experiments

- **E0. Clean baseline (false positives).** Run S2–S4 on a corpus of clean public
  models: torchvision (ResNet, MobileNet, ViT), GPT-2, BERT/DistilBERT, small bf16
  LLMs (Qwen2.5-0.5B, TinyLlama, SmolLM), GGUF quantizations. Survey ~200 Hugging Face
  models' safetensors without downloading whole files: the header gives exact byte
  ranges, so each tensor can be range-requested and sampled.
- **E1. Detection vs payload size.** Embed payloads of 1 KB–50 MB with each method
  (LSB-k substitution for k = 1..8 and 16, EvilModel byte replacement, spread
  across all layers or concentrated in one). Two payload kinds: random bytes (an
  encrypted payload, the hard case) and an unencrypted benign executable. No real
  malware is used anywhere. Metric: TPR at 1% FPR, FPR taken from E0.
- **E2. Cost to the model.** Accuracy drop (ImageNet validation subset for vision,
  WikiText-2 perplexity for language) for every E1 configuration. The headline plot
  overlays E1 and E2: the largest payload detection can't see, and what it costs.
- **E3. Adaptive attacker (break it).** An attacker who knows S3/S4: matches bit
  statistics, embeds only where the precision history permits, spreads thinly.
  Then fix what breaks.
- **E4. Differential (S5).** Real fine-tune/base pairs from Hugging Face: measure which
  tensors change in honest fine-tunes (full, LoRA-merged, head-only), then inject.
- **E5. Parity.** Same payload models through picklescan, ModelScan and a safetensors
  round trip, to show what each one says.

Every number is written to a results ledger with the command and commit that produced
it; tables in `docs/RESULTS.md` are generated, never hand-edited.

## Live demo (Remnant-style enforced claims)

A static page: drag a model in, get the report. The scanner core is ported to
TypeScript and runs in the browser, reading the file in slices (models are GBs).

- `Content-Security-Policy: connect-src 'none'`: the page technically cannot send the
  file anywhere. A browser test asserts that no request leaves the page.
- Parity: Python and TypeScript must produce byte-identical JSON reports on every
  fixture. Enforced in CI.
- The demo ships a few small poisoned fixtures to try it.

## Safety

Payloads are random bytes or benign programs. The extractor exists only inside the
test suite to prove embedded bytes are recoverable; no tool here runs extracted bytes.
The scanner never unpickles with real classes: `find_class` returns stubs only.

## Timeline

| Week | Work |
|---|---|
| 1 | Format parsers (safetensors, `.pt` zip + legacy, GGUF), S1, S2, CLI, tests |
| 2 | Payload embedding toolkit, S3, S4 v0, S5 |
| 3 | E0: clean corpus and Hugging Face range-request survey |
| 4 | E1 + E2, first plots |
| 5 | E3: adaptive attacker, fixes |
| 6 | E4 differential on real fine-tune pairs, E5 parity |
| 7 | Browser app: TypeScript core, parity tests, CSP, browser tests |
| 8 | Project page, FINDINGS, reproducibility page |
