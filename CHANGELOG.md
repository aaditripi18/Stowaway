# Changelog

Versions follow `MAJOR.MINOR.PATCH`; while the major version is 0, a minor
release may change detector rules and report fields. Every rule change is
pre-registered in [PLAN.md](PLAN.md) and validated on models committed before
their weights were read; the numbers are in [docs/RESULTS.md](docs/RESULTS.md).

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
