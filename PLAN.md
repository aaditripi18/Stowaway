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

## Phase 2 (written 2026-10-16, before the runs it governs)

### E4b. S5 on whole files

**Why.** E4 calibrated S5 on *sampled* tensors (the 6 largest roles, 32 KB each).
The first whole-file scan of a dev pair (`cybersectony/phishing-email-detection-distilbert_v2.4.1`
against `distilbert/distilbert-base-uncased`) flags 7 honest tensors: attention
key biases (which get no gradient, since softmax ignores a constant added to every
key score) and layer-norm weights. Their changes are small real updates whose
99th-percentile highest changed bit is 19-20, under the cutoff of 20.

**Candidate rule (S5b).** A payload written into the low k bits XORs roughly
uniform bits, so about half of the changed elements have bit k-1 as their highest
changed bit and none has a higher one. Honest small updates have a smooth tail
instead (on the dev pair, 0.3-0.8% of changed elements reach the highest bit; a
random payload at k = 8, 16, 20, 21 puts 47-53% there). S5b flags a tensor only
if its p99 highest changed bit is <= the cutoff (as now) **and** the share of
changed elements at the highest changed bit is >= a threshold T.

**Procedure.**
1. Dev: whole-file S5 and S5b on the 12 E4 dev pairs (already looked at). Choose T
   there and commit it, with the number of honest tensors each rule flags.
2. Fresh validation, used once: a new list of pairs from bases never used before
   (`results/ft_pairs_fresh.json`, frozen and committed before any weight is
   read), same selection rule as E4. For every pair, both rules, whole files:
   honest models with a false positive; and a random payload at k = 8, 16, 20, 21
   written (a) into the largest tensor that matches the base and (b) into the
   changed tensor with the smallest p99 bit (the low-gradient case).
3. **Decision:** S5b replaces S5 if, on the fresh batch, it has a false positive
   in at most 1 honest model, and catches every payload that S5 catches in case
   (a). Either way both rules' numbers are reported.

**Amendment (same day, before the dev run).** A payload written into a tensor
that training also changed (case b) can leave the highest changed bit set by the
training tail, which hides its ceiling from S5b. So dev also evaluates **S5c**:
flag a tensor if some bit j <= the cutoff is the highest changed bit of at least
a share T of the changed elements (a payload at k puts ~50% at j = k-1, whatever
else changed). Dev chooses between S5b and S5c, and T; the fresh batch then
judges the chosen rule once, by the decision rule above.

### E6. Wider E1/E2: nine models, five payload types, three seeds

E1/E2 covered two vision models, one seed and random payloads. E6 runs all nine
torchvision models of E0 (2 dev, 7 held out), payloads of five types (random;
zeros; English text; that text zlib-compressed; an executable-like blob: "MZ"
then random bytes), k in {1, 4, 8, 16}, 1 KB to 4 MB, seeds 0-2 for random
payloads (one seed for the rest). For each planted model it records single-file
detection (S3/S4, as in E1) and the bit statistics S5 reads (as in E4b, so any
S5 rule can be scored), and, on 1,000 Imagenette validation images, top-1
accuracy and **agreement** (share of images whose top-1 prediction is unchanged
from the clean model).

- **H5** Hiding is free: on every model, a random payload of up to 1 MB at k <= 8
  leaves >= 99% of top-1 predictions unchanged (all three seeds).
- **H6** S4's one-sided rule costs what it says: zero and text payloads are caught
  from the file alone in fewer cells than random payloads of the same size and k,
  summed over models; the base diff catches all of them wherever it catches random.

### E0d. S4 on small role groups

**Why.** S4 reads a tensor against its role peers when there are >= 3. With 3
peers the between-tensor spread (tau) is estimated near zero, so a small real
difference gives an extreme z: RegNet-Y-400MF's block-4 `proj` sits at 50.0% on
bit 0 against 51.2% for its three peers, z = -7.4 (FINDINGS, section 0c).

**Rule (S4b).** For a role reference with fewer than 6 peers, use
tau = max(tau of the peers, tau of the whole model's tensors of that dtype):
a handful of tensors cannot show less spread than the model has. Nothing else
changes.

**Validation.** Dev: the nine torchvision models (all have now been looked at).
Fresh, used once: a fourth batch of Hugging Face models (`results/hf_models_batch4.json`:
20 F32, 20 BF16, 20 F16 by the survey's rule, excluding every earlier batch),
frozen before any weight is read; S4 and S4b on identical samples. **Decision:**
S4b replaces S4 if, on the fresh batch, it flags no more models than S4 in any
dtype, and the E1b random-payload detection count per dtype drops by at most one
model.

**Dev result and amendment (same day, before the fresh batch is read).** S4b
leaves the RegNet flag in place (the model-wide spread is also small, so the
floor barely moves). Second candidate, **S4c**: a bit plane is tested only if
its reference is at least 0.02 from a fair coin (the same 0.02 S4 already uses
to decide that a model "carries a low-bit signature"): a reference that is
itself nearly a fair coin has no signature for a payload to erase. Dev compares
S4, S4b, S4c; the fresh batch then judges the chosen one by the decision rule
above.
