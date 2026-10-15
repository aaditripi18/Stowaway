# Findings (preliminary)

> Status: week 5 of 8 (S4 and S5 both revised and validated on held-out models).
> Two vision models end to end, single-file checks on samples of 120 public
> Hugging Face models (70 dev, 50 held out), and S5 calibrated on 20 real
> fine-tune/base pairs (dev + held out). One payload type (random bytes).
> Every number below is in [RESULTS.md](RESULTS.md), generated from the ledger
> with the commit that produced it. These are early results, not conclusions.

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
  of E1 was caught), **spread_all evades S4 at every size up to 1 MB**. Thinning
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

## 0c. S4 on 16-bit files: fixed for bf16 (validated on unseen models), not for fp16

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

So the fix holds up on unseen bf16 models (a 71% reduction), less well than the
dev set suggested, and does nothing for fp16. What is still flagged on the fresh
batch: tensors with no role peers in the sample, read against the whole model
(embeddings, `lm_head`, FP8 scales, MoE expert blocks: 7 of 13 flags); tensors
stored with *less* precision than the rest (a bit plane with mean exactly 0,
which a payload cannot produce but S4 still flags); and layer-0/3 norms of
heavily quantized MLX releases and AWQ scales. Detection of a random payload in
16-bit files stays low (2 / 19 bf16, 2–3 / 9 fp16), because their low bits are
mostly fair coins.

Practical reading: on bf16/fp16 files S4 is still not reliable enough to act on
alone; a clean result there means as little as on fair-coin float32 models.

The fresh batch is not a random sample: by the selection rule, many of its "bf16"
models are quantized releases (FP8, NVFP4, MLX 4–8 bit) in which only norms,
embeddings and scales are bf16, and four are MLX variants of one Qwen3.8-27B.

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
- No adaptive attacker yet (E3): one who knows S4 could embed only in tensors or
  bits without a signature, or preserve the signature.
- E2 measures accuracy; agreement of individual predictions with the clean model
  would be a more sensitive measure of cost.
- The survey tests detection on 64 KB samples, not whole files, and only the
  single-file checks; accuracy cost (E2) is measured only on the two vision
  models.
- S4 on fp16 files is not fixed (section 0c); S4 role peers need a better estimate of clean spread for small groups.
