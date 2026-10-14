# Findings (preliminary)

> Status: week 4 of 8. Two vision models end to end, plus single-file checks on
> samples of 70 public Hugging Face models. One payload type (random bytes).
> Every number below is in [RESULTS.md](RESULTS.md), generated from the ledger
> with the commit that produced it. These are early results, not conclusions.

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
(position embeddings in CLIP ViT-L/14 and ESM-2, a dense layer in XLM-R). S3
flags those, and from a single file it cannot tell such a tensor from one whose
low bits are filled with a random payload.

## 0c. S4 is not usable yet on files stored as bf16 or fp16

On files stored in 16-bit formats, S4 flagged clean tensors in 8 of 20 bf16 and
7 of 11 fp16 models, and saw a payload in only 3 of 30. The cause is a design
choice, not the data: S4 examines the "low 8 bits", which is the whole 7-bit bf16
mantissa and 8 of fp16's 10 bits, so it tests bits that encode the value
distribution itself (layer-norm weights near 1.0, router weights, AWQ scales).
Those account for 29 of the 40 flagged tensors in the survey.

The fix (examine only the low part of the mantissa, as for float32) will be
validated on a fresh batch of 16-bit models, not on these, which are now the
data the problem was found on.

Smaller patterns among the remaining false positives: tensors stored with *less*
precision than the rest of their model (a bit plane with mean exactly 0, e.g. a
sentence-transformers `Dense` layer). A random payload moves bits towards 0.5,
never to exactly 0, so these are harmless, but S4 still flags them.

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
tensors are fair coins. It is reported here and was not tuned away.

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
- S4 on bf16/fp16 files needs the fix in section 0c and a fresh validation set.
