# Findings (preliminary)

> Status: phase 2 (PLAN.md). S3, S4, S5 all revised and validated on held-out models;
> E5 compares Stowaway with picklescan, ModelScan and fickling.
> Two vision models end to end, single-file checks on samples of 120 public
> Hugging Face models (70 dev, 50 held out), and S5 calibrated on 20 real
> fine-tune/base pairs (dev + held out). One payload type (random bytes).
> Every number below is in [RESULTS.md](RESULTS.md), generated from the ledger
> with the commit that produced it. These are early results, not conclusions.

## Nine models, five payload types: hiding is free, and what gives a payload away is not its content (E6)

E6 widened E1/E2 to all nine torchvision models, five payload types (random;
zeros; English text; that text compressed; "MZ" plus random bytes), k = 1, 4, 8
and 16, 1 KB to 4 MB, and three seeds for random payloads, scoring top-1
agreement with the clean model on 1,000 Imagenette images.

**H5 supported: hiding is free.** On every model and seed, a random payload of up
to 1 MB in the low 8 bits leaves at least 99.8% of top-1 predictions unchanged
(seven of the nine models: 100.0%). Only k = 16 starts to show (ResNet-18: 98.9%
agreement at 4 MB).

**The base diff catches everything.** S5 (the adopted S5c rule) flags every one
of 216 seed-0 cells for random, zero and text payloads alike.

**H6 not supported, and the reason is instructive.** The prediction was that
S4's one-sided rule (a payload pulls bits toward a fair coin) would miss zero and
text payloads more often than random ones. From the file alone they were caught
*more* often: random 52, zeros 72, text 91 of 216 cells. Broken down by model:

| Models | random, compressed, "MZ"+random | zeros | text |
|---|---|---|---|
| the six whose low bits are already fair coins (ResNet-18, MobileNetV2, MNASNet, RegNet, ShuffleNet, SqueezeNet) | **0 caught**, every model | 5-8 of 21-27 per model | 3-8 per model |
| the three with a low-bit signature (MobileNetV3-Small/Large, EfficientNet-B0) | 14-19 of 24-25 per model | 5-13 | 13-20 |

The structured catches in fair-coin models are all large payloads (256 KB to
4 MB), all by S4, and they come from the payload contaminating S4's reference: a
large payload of zeros or text gives most tensors of a role the same low bits,
so the role's reference moves away from a fair coin, and the one tensor the
payload only partly filled sits nearer 0.5 than its peers. That is a real catch
of a tensor carrying the payload, but it comes from the payload's size and
layout, not from recognising its content. What content does not change: random,
compressed and "MZ"+random payloads behave identically (0 catches in every
fair-coin model), so encrypting or compressing a payload buys an attacker
nothing that a fair-coin model does not already give.

**The cost of S4c.** Re-scoring E6's planted models with S4c (E0d) turns 14 of the
302 caught cells (994 planted) in the six re-scored models into misses (ResNet-18 17 -> 14,
ShuffleNet 20 -> 16): planted tensors whose reference was within 0.02 of a fair
coin, exactly where S4c stops testing. E0d's fresh batch showed no such loss on
its random payloads; E6's wider grid does, and it is reported here.

## The base diff passed on samples and failed on whole files (E4b)

E4 calibrated S5 on sampled tensors (32 KB from the six largest roles) and
validated it with 0 false positives on 15 held-out fine-tunes. The first scan of
a whole file contradicted that: `cybersectony/phishing-email-detection-distilbert_v2.4.1`,
an honest fine-tune of `distilbert-base-uncased`, had 7 tensors flagged as
low-bit payloads. Run on all 12 dev pairs, whole files, the E4 rule flagged 129
honest tensors in **8 of 12** fine-tunes.

The flagged tensors are the ones that barely train. Attention key biases get no
gradient (softmax ignores a constant added to every key score) and LayerNorm
weights move little, so a fine-tune changes them only in the low bits, which is
exactly what the E4 rule took to be a payload. Sampling the largest roles had
never looked at them.

Two candidate fixes, both written into PLAN.md before any fresh pair was read:

- **Weight decay?** No. The changes shrink magnitudes only about half the time,
  like a random payload; this explanation was tested and rejected.
- **How a payload changes bits.** A payload written into the low k bits is XORed
  with roughly random original bits, so bit k-1 is the highest changed bit of
  about half the changed elements, whatever the payload (random 47-53% on the dev
  tensors), while honest small updates spread smoothly (0.3-0.8% at the top bit).
  S5b tested this at the top changed bit; S5c at the most common bit at or below
  the cutoff, which also works when training changed some higher bits.

On dev, S5b removed every false positive but missed nearly every payload hidden
in a low-gradient tensor; S5c at a fixed threshold of 0.4 had no false positives
but honest shares reached 0.382 in 768-element tensors, where sampling noise is
about 0.018. The chosen threshold is three binomial standard deviations below
one half, T(n) = 0.5 - 1.5/sqrt(n) for n changed elements (0.406 at 256, 0.446 at
768).

Run once on 20 fresh pairs from four new bases (MiniLM, ALBERT, DistilRoBERTa,
BERT-cased; 2,114 changed shared tensors):

| | Honest fine-tunes flagged | Honest tensors flagged | Payload in a base-matching tensor (k = 8/16/20/21) | Payload in a low-gradient tensor (k = 8/16/20/21) |
|---|--:|--:|---|---|
| S5 (E4 rule) | 16 / 20 | 185 | 9/9, 9/9, 9/9, 9/9 | 16/20 at every k (meaningless with these false positives) |
| **S5c, T(n)** | **0 / 20** | **0** | **9/9, 9/9, 9/9, 9/9** | 1/20, 8/20, 16/20, 18/20 |
| S5c, T = 0.4 (rejected on dev) | 2 / 20 | 4 | 9/9 at every k | 2/20, 13/20, 18/20, 18/20 |

By its rule S5c replaced S5. What it still cannot do: see a small-k payload
inside a tensor the fine-tune genuinely trained (1 of 20 at k = 8). The browser
and the CLI run the same rule, byte-identical in the parity tests.

## S4 on a fourth fresh batch (E0d)

The one remaining dev false positive of S4 (RegNet-Y-400MF's block-4 `proj`,
three peers) came from an under-estimated spread. Flooring the spread at the
model's (S4b) did not remove it; testing a bit plane only when its reference is
at least 0.02 from a fair coin (S4c, the same 0.02 S4 uses to call a model's low
bits a signature) did. On 60 unseen models S4c changed nothing: models with a
false positive were F32 2 / 20, BF16 1 / 20, F16 0 / 20 for S4 and S4c alike, with
the same payload detections (8, 2 and 3 of 20). It was adopted by its rule, and
these are S4's honest numbers on unseen files: single-file statistics are a hint,
reported as `suspicious`, not proof.

## The scanners people already use: every weight and structure payload passes them (E5)

E5 put one corpus of 21 files through picklescan 1.0.5, ModelScan 0.8.8 and
fickling 0.1.12 (all current releases on 2026-10-16, default settings), and
through Stowaway. The corpus: 6 clean files, 8 code payloads (a pickle global
that runs on load; inert arguments), 4 weight payloads (64 KB of random bytes in
the low 8 bits of ResNet-18, as trained and as if trained in bf16) and 3
structure payloads (an executable-looking blob after the end of a safetensors
or GGUF file, or as an extra entry in a `.pt` zip).

| Payloads flagged | picklescan | ModelScan | fickling (ML allowlist) | Stowaway | Stowaway + base |
|---|--:|--:|--:|--:|--:|
| code (8) | 8 | **6** | 7 | 8 | |
| weight (4) | 0 | 0 | 0 | 2 | 4 |
| structure (3) | 0 | 0 | 0 | 3 | |

**H1 holds.** Stowaway's S1 flags the same 8 / 8 code payloads as picklescan,
and every weight-payload model passes picklescan. Converting each `.pt` to
safetensors the recommended way (`torch.load(weights_only=True)`, then
`safetensors.torch.save_file`) refused all 6 code-payload files without running
anything, but carried both weight payloads across bit for bit; the 64 KB is
still extractable from the "safe" file. Conversion also silently drops the extra
zip entry, so for structure payloads it is a fix, not a hiding place.

**ModelScan misses 2 of 8 code payloads, both legacy-format `.pt` files**
(`torch.save(..., _use_new_zipfile_serialization=False)`, the format before
PyTorch 1.6, still loaded today). ModelScan reads the magic number and then
only one more pickle (`scan_pytorch` calls
`scan_pickle_bytes(multiple_pickles=False)`), but the model object is the
fourth pickle in the file. `exec` and `os.system` there are both missed. This
is already publicly reported (several proof-of-concept repositories on
Hugging Face describe the same truncation); E5 reproduces it independently.

**False positives.** fickling's default check flags all 3 clean pickle files
(any import outside the standard library, `torch._utils` included, is "likely
unsafe"), so on PyTorch files its default verdict carries no information; its
ML allowlist analysis flags none and is the column used above. Stowaway flags 1
of 6 clean files: a plain pickle holding a numpy array is "suspicious" because
`numpy._core.multiarray._reconstruct` is not on its allowlist. That is the same
call PyTorch makes: `torch.load(weights_only=True)` refuses this global by
default. We left it as is. fickling errors on the protocol-0 pickle
(`NotImplementedError: TODO: Add support for Opcode FLOAT`).

**What none of the existing scanners look at.** Weight payloads in the bf16-origin
model are caught by S3 from the file alone; in the as-trained ResNet-18 only the
base diff (S5) catches them, as in E1. Structure payloads are caught by S2 in
all three formats. The three existing scanners read pickles only, so for
`.safetensors` and `.gguf` they report clean or skip the file. The corpus is
small and built by us, so this shows which kinds of payload each tool can see,
not detection rates in the wild.

## An adaptive placement evades the single-file checks; the base diff is the backstop (E3)

E3 stress-tests S3/S4 the way a defence should be: assume the person hiding the
data knows how the checks work and places random bytes where the checks are
weakest, then measure what gets through. Three placements, from the weaknesses
found earlier, all using ordinary parameters of the embedder (which tensors, how
thinly spread):

- **spread_all** — the low bits of every float tensor, one weight in 64, so each
  tensor's bit-plane means barely move.
- **role_fill** — fill every tensor of one role, so the peers S4 reads a tensor
  against are changed too.
- **fair_only** — only tensors whose low bits are already fair coins, where S4
  has no signal.

Result on the two vision models:

- On **MobileNetV3-Small** (whose low bits carry a signature, so the naive attack
  of E1 was caught), **spread_all evades S4 at every size tested** (1, 4 and 16 KB:
  at 1 weight in 64 with k = 4, the model holds 19.5 KB; an earlier draft
  of this page wrongly said "up to 1 MB"). Thinning
  the payload across 1 weight in 64 keeps every bit-plane mean inside the z = 6
  threshold. This is a real evasion of the single-file checks on a model where
  the obvious attack failed.
- **role_fill** is a weaker evasion than expected: S4 still catches it from 4 KB,
  because filling a role shifts its peers together but the model-wide fallback and
  the remaining roles still expose it.
- On **ResNet-18** (already fair coins) every placement evades single-file, as the
  naive attack already did — there is simply no single-file signal to begin with.

In every case the **base diff (S5) catches the payload**, with one corner: a 1 KB
payload spread at 1:64 over MobileNetV3 touches ~98 tensors with fewer than the
256 changed elements per tensor S5 requires, so S5 skips them and misses it
(caught from 4 KB up). That per-tensor floor is a deliberate guard against noise;
lowering it needs its own held-out validation, so it is recorded, not changed.

The headline holds and sharpens: single-file statistics have a fundamental blind
spot — a payload thin enough moves no statistic far enough — and the comparison
against a trusted base is the check that does not depend on where in the file the
bytes go. The scanner's own output already says as much: on a clean single-file
scan, "no finding" is not "no payload".

## S5 recalibrated: honest fine-tuning never hides in the low bits (E4)

Week 3 set S5's cutoff — how low a tensor's changes must stay to look like a
payload rather than training — at half the mantissa (11 of 23 float32 bits), by
guess. A payload at k = 16 changed bits above it and was missed on both vision
models.

E4 measured the real boundary. For every tensor a fine-tune changed, it took
base XOR fine-tune and recorded the highest mantissa bit that differs. Across
**189 changed tensors in 8 dev fine-tunes** (BERT, DistilBERT, GPT-2 fine-tunes),
the 99th-percentile highest changed bit was the **top mantissa bit (22) in every
single tensor**: honest fine-tuning moves whole magnitudes, never just low bits.

S5 now flags a tensor whose changes reach only below bit (mantissa − 3): 20 bits
for float32, 7 for fp16, 4 for bf16. Validated once on **15 held-out pairs**
(RoBERTa, XLM-R, ViT fine-tunes, chosen and committed before their weights were
read):

- **0 honest false positives.**
- A payload in a tensor that should match the base is caught at k = 8, 16, 20, 21
  (10 / 10 pairs with sampled weights), and missed at k = 22 — above the cutoff,
  and a 22-bit payload is a change so large it is not plausibly hidden.
- E1 rerun confirms the week-3 k = 16 blind spot is closed on both vision models
  at every payload size.

**A real limit.** S5 only catches a payload in a tensor that otherwise matches
the base. A payload hidden inside a tensor the fine-tune genuinely trained is
camouflaged — the tensor already differs from the base across all bits — and was
caught in 0 / 10 held-out pairs at every k. So S5 defends the frozen parts of a
claimed fine-tune (and any model passed off as base + payload), not the trained
parts. A distributor who confines a payload to tensors they did train, and
declares a plausible amount of fine-tuning, defeats S5; catching that needs the
single-file checks, which for most float32 models (section 0) see nothing.

**Lineage aside.** Selecting these pairs surfaced that many top "fine-tunes" of a
base are not: several declare `base_model: X` but are distilled or resized to a
different hidden size (e.g. a 128-wide model citing bert-base-uncased, which is
768-wide). They were excluded from E4 by a parameter-count gate. That labels are
this unreliable is the premise of roadmap project #2.

## 0. Across 39 popular float32 models, the file alone exposes a payload in 16 and says nothing about 23

E0b/E1b sampled up to 12 tensors (64 KB each) from the 40 most-downloaded
non-gated float32 safetensors models (39 had testable samples) and filled the low
bits of one tensor with random bytes:

| What the weights look like | Models | Payload flagged from the file alone (k = 1, 4, 8) |
|---|--:|---|
| float32 file carrying only fp16 or bf16 precision | 14 | 14 / 14 at every k |
| float32 with a low-bit signature | 2 | 2 / 2 at every k |
| float32 with fair-coin low bits | 23 | **0 / 23** at every k |

So for 59% of these models, a clean single-file scan certifies nothing about the
weights; only a comparison with a trusted base (S5) can. The list is dominated
by BERT-family and sentence-embedding models because those are the most
downloaded; it is frozen in `results/hf_models.json`.

## 0b. 36% of "float32" uploads are really half precision, and that is where S3 works

14 of the 39 float32 files hold weights with only fp16 (13) or bf16 (1)
precision: the low 13 or 16 bits of every value are zero, as when a model
trained in half precision is saved with `.float()`. S3 catches a payload in all
of them (E1b).

E1c tests S3 directly on real bf16 weights: the samples of the 19 surveyed bf16
models, widened to float32 exactly as `.float()` does (checked bit for bit
against torch). S3 identified bf16 precision in 19 / 19, raised no false
positive in any of them, and flagged a payload as small as 1 KB in 19 / 19 at
k = 1, 4 and 8. Hypothesis H2 holds on real weights.

Its limit: three of the 14 half-precision models keep one tensor in full float32
(position embeddings in CLIP ViT-L/14 and ESM-2, a dense layer in XLM-R), and in
the fresh batch DeepSeek-V3.2 keeps its FP8 dequantization scales
(`weight_scale_inv`) in full float32 next to bf16-precision weights. S3 flags
those, and from a single file it cannot tell such a tensor from one whose low bits
are filled with a random payload. (With role sampling these standalone tensors are
rarely sampled, so the S3 false-positive count in the dev rerun dropped to zero;
that is a sampling effect, not a change to S3.)

## 0c. S4 on 16-bit files: fixed and validated on unseen models

The week-4 S4 flagged clean tensors in many models stored as bf16 or fp16. Two
causes, both found on the week-4 models (now the dev set):

- It examined the "low 8 bits", which is the whole 7-bit bf16 mantissa, so it
  tested bits that encode the value distribution itself (layer-norm weights near
  1.0 span only a few bf16 steps).
- It read every tensor against the whole model, but different kinds of tensor
  carry different low-bit signatures: bf16 routers and embeddings sit at 50% on
  bit 0 where their models sit at 39–44%, which is exactly what a payload looks
  like.

The current S4 examines the same low fraction of the mantissa in every format
(8 of float32's 23 bits, unchanged; 3 of fp16's 10; 2 of bf16's 7) and reads each
tensor against the other tensors of its role (same name without layer numbers,
same number of dimensions), falling back to the whole model when a role has
fewer than three peers. The survey now samples 4 layers of each of the 6 largest
roles so that peers exist.

The rule was frozen (commits edb3957, 7fd9378) and then run once on a fresh batch
of 50 models chosen by the same rule and committed before any of their weights
were read (`results/hf_models_batch2.json`). Models with an S4 false positive,
old rule against new on identical samples (E0c):

| Stored as | Dev batch: old → new | **Fresh batch: old → new** |
|---|---|---|
| bf16 | 9 → 0 of 20 | **14 → 4 of 20** |
| fp16 | 6 → 2 of 10 | **5 → 5 of 10** |
| float32 | 1 → 1 of 40 | **2 → 2 of 20** |

That first revision fixed bf16 (14 → 4 of 20 on a fresh batch) but did nothing
for fp16 (5 → 5 of 10). The flags that remained shared one shape: almost all were
bit planes *further* from a fair coin than their reference — a bit stuck at 0 in
a quantized embedding or `lm_head`, a norm with fewer than half its low bits set.
A random payload can never produce that: it only adds entropy, pushing a bit
plane *towards* 0.5.

So the current S4 (commit 67404c6) flags a bit plane only when it is *nearer* 0.5
than its reference, and skips quantization-scale tensors (AWQ `.scales`, FP8
`weight_scale_inv`), which are not weights and follow the quantization grid. This
loses detection of constant, zero or text payloads — they bias a bit *away* from
0.5, exactly like low precision, and cannot be told apart from one file — which
is left to the base diff (S5).

Validated once on a **third fresh batch** of 40 16-bit models (20 bf16, 20 fp16),
chosen by the same rule and committed before their weights were read
(`results/hf_models_batch3.json`). Models with an S4 false positive, the first
revision against the current rule on identical samples (E0c):

| Stored as | dev batches | batch 2 | **batch 3 (held out): first → current** |
|---|---|---|---|
| bf16 | 10 → 2 of 40* | 14 → 4 of 20 | **5 → 1 of 20** |
| fp16 | — | 5 → 5 of 10 | **1 → 0 of 20** |

<sub>*dev bf16/fp16 recount under the current rule; the per-batch comparison is in RESULTS.md.</sub>

So on fully unseen models the current rule flags **1 of 20 bf16 and 0 of 20 fp16**.
The one bf16 flag is an NVFP4 4-bit-quantized release. The batch-3 fp16 models are
mostly AWQ/GPTQ, which the first revision already passed, so the larger fp16 gain
shows on the dev batches (10 → 2 of 40), not here.

Detection of a random payload in 16-bit files stays low (≈2 / 20 bf16, 2 / 20
fp16): their low bits are mostly fair coins, where no single-file check has a
signal. So a clean S4 on a 16-bit file still means little — the base diff is the
check to rely on — but S4 no longer *cries wolf* on clean quantized models.

The batches are not random samples: by the selection rule, many "bf16" and "fp16"
models are quantized releases (FP8, NVFP4, AWQ, GPTQ, MLX) in which only norms,
embeddings and scales keep 16-bit precision.

Two weaknesses of the role rule, found on dev models after it was written:

- **Small groups.** In RegNet-Y-400MF (torchvision, held out), the block-4 `proj`
  tensor sits at 50.0% on bit 0 against 51.2% for its three peers. With three
  peers the spread between clean tensors is estimated near zero, so a 1.2-point
  difference in a large tensor gives z = −7.4.
- **A payload that covers a whole role.** In MobileNetV3-Small, payloads large
  enough to fill most tensors of a role poison the peers too, and S4 misses some
  of them (e.g. k = 1 at 64 KB, 176 tensors written) where the week-3 rule caught
  them. This is the first lead for the adaptive attacker (E3).

## 1. A 4 MB payload can sit in ResNet-18 at no measurable cost and pass every single-file check

Random bytes (the hard case: an encrypted payload) written into the low 8 bits of
ResNet-18's float32 weights:

- **E2:** up to 4 MB leaves top-1 accuracy *exactly* unchanged on 500 Imagenette
  images (66.8% before and after).
- **E1:** S3 and S4 miss it at every size tested, 1 KB to 1 MB at k = 1, 4 and 8.
  So does every code scanner, since there is no code.
- **E1:** S5, a bit-level diff against the clean base model, catches all of them.

4 MB is more than enough for a working program. For a model like ResNet-18,
scanning the file alone cannot rule out a weight payload; comparing against a
trusted base can.

## 2. Whether a single file is enough depends on the model

ResNet-18's low mantissa bits look like fair coins. The official MobileNetV3-Small
weights do not: bit 0 is set in ~67% of values and bit 1 in ~37%, consistently in
every tensor. We do not know which step of how those weights were produced leaves
this signature; we only measure it.

The signature makes hiding harder. A random payload pulls the bits it touches back
towards 50%, so on MobileNetV3-Small S4 detects every payload tested from the file
alone, down to 1 KB at k = 1 (the lowest single bit of each weight).

So hypothesis H3 ("random low-bit payloads cannot be seen from one fp32 file") is
true for some models and false for others. The survey (section 0) puts numbers on
it: a signature in 2 of 39 popular float32 models, fair-coin bits in 23.

## 3. The fair-coin assumption produced false alarms; the fix is post hoc

The first S4 assumed low bits are fair coins and raised 150 false positives on the
unmodified MobileNetV3-Small. S4 now tests each tensor against the model's own
median signature, allowing for the spread between clean tensors, and on bits where
the model has a signature it only counts movement towards 50%.

That rule was designed after seeing MobileNetV3-Small, so it was then run on seven
torchvision models that had not been looked at:

| | Models | False positives |
|---|---|---|
| dev (used to design S4) | 2 | 0 |
| held out | 7 | 1 tensor in 1 model |

The one held-out false positive is ShuffleNetV2's final classifier bias
(`fc.bias`, 1,000 values): bit 0 is set in 35% of them, in a model whose other
tensors are fair coins. It is reported here and was not tuned away. With the week-4 role rule (section 0c), ResNet-18 is
clean again and the held-out count is 2 tensors in 2 models (ShuffleNetV2's
`fc.bias`, RegNet-Y-400MF's block-4 `proj`).

## 4. S5 has a blind spot at k = 16

S5 flags a tensor when its changes are confined to the lowest 11 mantissa bits
(half the mantissa), because a fine-tune changes weights by more than that. A
payload in the low 16 bits changes bits above that cutoff and is missed at every
size, on both models (E1). At k = 16 the accuracy change was at most 1.2 points
either way, within sampling noise for 500 images (E2).

The cutoff has to separate payloads from honest fine-tunes, so it will be set from
real fine-tune/base pairs in E4 rather than adjusted to this result.

## Not yet tested

- Only two models in E1/E2, one seed, random payloads only.
- E3 tested placements from ordinary embed parameters only; an attacker who
  preserves each tensor's bit signature exactly was not tested.
- E2 measures accuracy; agreement of individual predictions with the clean model
  would be a more sensitive measure of cost.
- The survey tests detection on 64 KB samples, not whole files, and only the
  single-file checks; accuracy cost (E2) is measured only on the two vision
  models.
- S4 role peers still need a better estimate of clean spread for very small groups (RegNet case); the two residual bf16 flags are extreme sub-4-bit quantized models.
