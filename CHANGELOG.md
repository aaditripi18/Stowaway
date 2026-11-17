# Changelog

Versions follow `MAJOR.MINOR.PATCH`. Before 1.0.0, a minor release could change
detector rules and report fields; from 1.0.0, a change to a report field is a
major release and a change to a detector rule at least a minor one. Every rule change is
pre-registered in [PLAN.md](PLAN.md) and validated on models committed before
their weights were read; the numbers are in [docs/RESULTS.md](docs/RESULTS.md).

## 1.0.0 — 2026-11-17

### Detection (both the Python tool and the browser build, with parity fixtures)
- **S4w, windowed S4**: each 32,768-element window of a tensor is tested
  against the tensor's other windows, so a payload that fills part of a large
  tensor is no longer diluted by the rest. Findings use the new code
  `lowbit-window-anomaly` (E11: whole file and sample agree on 163 / 168 planted
  cells; no clean model of 65 gains a flag).
- **S5 judges every changed tensor**: the 256-changed-element floor is removed;
  S5c's threshold already scales with the number of changes (E12: a thin payload
  caught in 18 / 18 fresh models, 0 / 18 with the floor).
- **S5d**: S5 also requires that at most 2% of changed elements change above
  the payload's top bit, so LayerNorm gains and key biases a fine-tune shifted
  are no longer flagged. S5 findings gain an `above` detail (E13: 48 / 48
  base-matching catches kept, the thin payload in 30 / 30 models).

- **Is the base close enough?** With a base, S5 reports the median cosine of the
  aligned tensors and how many are identical (`base-closeness`, info), and below
  0.9 warns that S5 can only see the identical tensors (`base-not-close`, medium).
  E15: fired on 9 / 9 independent same-architecture bases, on no confirmed parent
  S5 can use.
- **S4x**: on whole files, S4's whole-tensor test is skipped for a tensor S4w
  already tests window by window (at least three windows, not an embedding table
  or a quantisation scale); the tensor still serves as a peer. E17: every one of
  20 fresh catches kept, models with a false flag 5 -> 2.

### Evidence
- E11, E12, E13: see docs/FINDINGS.md, phase 4. H13-H18 supported.
- E14, E15, E17: see docs/FINDINGS.md, phase 5. H19, H20, H23 and H24
  supported; E14 found no safe rule for never-trained tensors (not adopted).

## 0.3.0 — 2026-11-03

### Removal
- **`stowaway sanitize`**: rounds every float32 weight at mantissa bit 8
  (`--bits`) and writes a new file; a low-bit payload cannot survive it. On nine
  image models and a text classifier, 8 bits left 100.00% of predictions
  unchanged and 0 bytes of a planted 64 KB payload (E9).

### Scanning
- S5's dictionary entry point aligns tensor names exactly as `scan --base`
  does; before, a base whose names differed by a wrapper prefix was compared on
  identical names only (found in E8).
- The Hub client waits out rate limits (HTTP 429) and transient server errors,
  respecting `Retry-After`, instead of failing.

### Evidence
- E8 reference discovery, E9 sanitising, E10 whole files and a random sample:
  see docs/FINDINGS.md, phase 3. H7, H8, H10, H11 and H12 not supported; H7b and
  H9 supported.

## 0.2.0 — 2026-10-16

### Scanning
- Files of any size, in the CLI and the browser: both read one tensor (and its
  base counterpart) at a time. A 3.19 GiB model: 0.375 GiB peak (Python), 0.17 GiB
  (browser core), identical reports.
- `stowaway scan hf:org/name[@revision]` and `--base hf:…` / `--base declared`:
  Hub files pinned to a commit, checked against the Hub's SHA-256, truncated
  transfers resumed; the report records the commit.
- S5 pairs tensors with the base across wrapper prefixes and old LayerNorm names,
  and accepts a base split over several files.
- **S5c replaces the E4 S5 rule** (E4b): the old rule flagged 16 of 20 fresh
  honest fine-tunes scanned as whole files; S5c flags 0 and still catches every
  payload in a tensor identical to the base. Its finding message and `detail`
  changed (`bit`, `share`, `threshold`).
- S4c: a bit plane is tested only if its reference is at least 0.02 from a fair
  coin (E0d; removes a false positive on small role groups).
- Exit status 3 when a scan cannot run.

### Browser
- Legacy (pre-1.6) `.pt` files and deflate-compressed zip entries, byte-identical
  to Python.
- The scan runs in a worker started from a `blob:` URL, under the page's
  `connect-src 'none'`.
- A limits page: plant random bytes in a model (bundled Apache-2.0 BERT or your
  own file), sweep size and depth, download the planted copy.

### Project
- CI on Python 3.10–3.13, Linux, Windows and macOS; a numpy-only install job; the
  browser suite; the Docker image built and run.
- `docs/API.md`: the supported Python API, the report schema and every finding
  code. `scripts/site_facts.py`: every number on the site checked against the
  ledger.
- Experiments E5 (versus picklescan, ModelScan, fickling), E0d, E4b and E6.

## 0.1.0 — 2026-10-15

First release: format parsers (safetensors, PyTorch zip and legacy, pickle,
GGUF), checks S1–S5, the CLI, experiments E0–E4, and the browser scanner for
safetensors, GGUF and PyTorch zip files.
