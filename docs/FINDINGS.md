# Findings (preliminary)

> Status: week 3 of 8. Two vision models, one seed, one payload type (random
> bytes). Every number below is in [RESULTS.md](RESULTS.md), generated from the
> ledger with the commit that produced it. These are early results, not
> conclusions.

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
true for some models and false for others. The next step is to measure how common
such signatures are across many public models (week 3–4 survey).

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
- Language models and bf16-trained models (where S3 applies) are not in the
  experiments yet.
