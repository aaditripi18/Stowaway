# Stitch prompts for the Stowaway project page

Use Stitch in Web mode. Paste prompt 1 and generate. Then paste prompts 2–5 one
at a time as refinements of the same design. Export the HTML/Tailwind at the end
and upload it to the session. Numbers get wired to the site's data afterwards, so
don't worry if Stitch rounds or tweaks a figure.

The visual system is the one the scanner already uses, so the landing page and
the scanner feel like one site: a near-black technical background, off-white text,
Hanken Grotesk for prose and JetBrains Mono for numbers / tensor names / code, one
cyan signal colour (#38bdf8) used only where data is *hidden* or *found*, and red
(#ffb4ab) reserved for "dangerous". Thin hairline rules, lots of whitespace,
~1100px content width, dark by default.

---

## Prompt 1: the page (paste first)

A one-page website for "Stowaway", a security research tool that finds data hidden
inside AI model files — payloads buried in the low bits of the weights, bytes the
file format doesn't account for, and unsafe code in pickle data — and proves what a
scanner that only reads one file cannot see.

Feel: a forensics lab crossed with an editorial research report. Calm, precise,
technical, trustworthy. Think a well-designed security write-up, not a startup
landing page. No gradients, no 3D blobs, no stock imagery, no hype.

Look: near-black background (#0f131c), off-white text (#dfe2ef), hairline dividers,
Hanken Grotesk for headings and prose, JetBrains Mono for numbers, tensor names and
code. One signal colour, cyan #38bdf8, used only where something is hidden or
found. Red #ffb4ab only for a "dangerous" verdict. 1100px content width. A light
mode is optional.

Hero, the centrepiece: on the left, a large headline "What can hide in a model's
weights that no scanner sees?" and a short line under it: "Stowaway inspects
.safetensors, .pt and .gguf files for payloads in the weights, unaccounted bytes
and unsafe pickle code — entirely in your browser, nothing uploaded — and shows
that reading one file is not enough." Three buttons: "Read the findings",
"Scan a model" (primary, cyan), and "Reproduce it". On the right, a custom
illustration built from UI elements: a grid of monospace 32-bit float values shown
as hex, like a weight tensor; the lowest byte of each value is tinted cyan, and
the tinted bytes together spell out a faint hidden message. Under it, a huge mono
number "4 MB" with the caption "of hidden data carried in ResNet-18 at zero
accuracy cost — invisible to every single-file scanner."

Below the hero, a slim row of three stats separated by hairlines: "5 checks ·
136 tests", "39 popular float32 models surveyed", "16 of 39 leak a payload from the
file alone; 23 reveal nothing".

Then sections, each with a small mono section number (01, 02, …) and a short title:
The problem · What it checks · The experiments · What one file can't see · Four
hypotheses · What we learned · Validated on unseen models · Reproduce ·
Limitations. Sticky top bar with the "STOWAWAY" wordmark in mono and links
Findings, Checks, Scan, Reproduce, GitHub.

---

## Prompt 2: the evidence sections (paste second)

Design "The experiments" and "What one file can't see" in detail.

The experiments: a short lead line "A payload is just numbers. It survives
conversion to a 'safe' format, costs the model nothing, and no code scanner looks
at it." Then a clean full-width table with hairline row dividers, right-aligned
mono numbers, no zebra stripes. Title it "Where a random payload hides, by what the
model's weights look like." Columns: Model group, Models, Seen from the file alone.
Rows: "float32 file, really fp16/bf16 precision · 14 · 14 / 14, every size"; "float32
with a low-bit signature · 2 · 2 / 2, every size"; "float32, fair-coin low bits · 23
· 0 / 23, every size". The last row's "0 / 23" gets a soft cyan background pill —
this is the point. A one-line caption: "For 23 of 39 popular models, a clean
single-file scan certifies nothing about the weights."

What one file can't see: a horizontal comparison drawn with simple bars/markers for
two models, ResNet-18 and MobileNetV3-Small, across payload sizes 1 KB to 1 MB.
For each, three lanes: "Single-file checks (S3/S4)", "Base-model diff (S5)",
"Accuracy cost". Show ResNet-18 as: single-file = missed at every size (muted),
S5 = caught at every size (cyan), accuracy cost = 0.0 points. MobileNetV3-Small as:
single-file = caught at every size (cyan), S5 = caught, accuracy cost = 0.0 points.
Above it, a large takeaway: "A 4 MB payload in ResNet-18 changes no prediction on
500 test images and passes every single-file check. Only a bit-for-bit diff against
the base model catches it."

---

## Prompt 3: hypotheses and findings (paste third)

Design "Four hypotheses" as four compact horizontal cards stacked vertically, each
with a mono ID chip (H1–H4), the hypothesis in one line, a verdict pill on the
right (cyan "Supported", grey "Partly / model-dependent"), and a muted one-line
piece of evidence:
H1 — "A payload hides in the weights as numbers, survives conversion to
.safetensors, and passes every code scanner." Supported (code scanners see
nothing; the bytes are just weights).
H2 — "In a model stored at higher precision than it was trained in, a payload of
1 KB or more is caught from the file alone, with no false positives." Supported
(S3 identified bf16 origin in 19 / 19 models, 0 false positives, caught a 1 KB
payload in all).
H3 — "In a genuinely float32-trained model, an encrypted low-bit payload cannot be
seen from one file." Model-dependent (true for 23 of 39 models, false for the 2
with a low-bit signature).
H4 — "A diff against the claimed base model catches a payload in any tensor the
fine-tune left unchanged." Supported, with a limit (a payload inside a genuinely
re-trained tensor is camouflaged).
Add a small note: "Hypotheses were written before the experiments ran."

Design "What we learned" as a 2×2 grid of editorial cards, each with a bold
headline and two sentences:
"A clean scan is not a clean model." For most popular float32 models, single-file
statistics have no signal; a payload thin enough moves no number far enough. The
only reliable check needs the trusted base model to diff against.
"Whether one file is enough depends on the model." Some models' low bits are fair
coins that hide anything; others carry a signature a payload disturbs. We measure
which, we don't assume.
"A third of 'float32' uploads are really half precision." 14 of 39 top models are
saved as float32 but carry only fp16 or bf16 weights, leaving a block of zero low
bits — exactly where a payload would go, and exactly where S3 can see it.
"The strong check has a blind spot too." A payload confined to tensors the
distributor genuinely trained is camouflaged by the training, so even the base diff
can miss it. We report the limit rather than hide it.

---

## Prompt 4: trust and footer (paste fourth)

Design "Validated on unseen models" as three large mono figures with captions, in a
row separated by hairlines: "0 false positives — S5 on 15 held-out fine-tune/base
pairs"; "5 → 1 of 20 — S4 false-positive models on a fresh bf16 batch, before and
after the fix"; "189 / 189 — fine-tuned tensors whose changes reach the top
mantissa bit, which is how S5's cutoff was set." Under them a muted line: "Every
detector was frozen and then run once on a batch of models committed before their
weights were read, so the numbers are not tuned on what you see here."

Design "Reproduce" as a dark terminal-style code block with a copy button,
containing:
python -m stowaway scan model.safetensors
python -m stowaway scan model.pt --base base.pt
python scripts/run_hf_survey.py --list hf_models.json
python scripts/report.py
with a caption "Laptop CPU only · nothing is trained, models are downloaded and run
forward · every number comes from an append-only ledger · 136 tests (106 Python,
30 in the browser build, byte-for-byte identical reports)."

Design "Limitations" as a quiet, muted list of short lines, deliberately
understated: single-file statistics cannot see a small or thinly-spread encrypted
payload in a fair-coin model; the base diff needs the real base model and misses a
payload inside a re-trained tensor; the model survey is weighted to the most
downloaded models, not a random sample; legacy (pre-1.6) torch files and compressed
zip entries are read by the command-line tool, not the browser.

Footer: the mono "STOWAWAY" wordmark, the line "Every claim on this page is a test
you can run.", and links GitHub, Findings, Reproduce.

---

## Prompt 5: the scanner page (a second screen, same design system)

Add a second page in the same style: "Scan a model file". A working version of this
already exists; design the shell so the two match, and keep these elements and ids
so the real scanner can be wired in.

Top: title "Scan a model file" and a reassurance line with a small lock icon: "Your
file never leaves this page. A content-security policy technically blocks the page
from sending data anywhere."

A large drop zone with a dashed hairline border: "Drop a .safetensors, .pt or .gguf
file to scan it", with a "Choose a file" button. Under it a smaller, optional
second slot: "Compare against a base model (optional) — diff the weights bit for
bit (the S5 check)."

After a scan, a result dashboard: a verdict banner that is one of three looks —
Clean (calm green), Suspicious (amber), Dangerous (red) — with one plain-language
line. A row of fact tiles: file name, format, size, tensor count, and a truncated
SHA-256 with a copy button. A findings list where each finding is a card with a
severity chip, a mono tag for the check that raised it (S1–S5), a one-line message,
and an expandable area with key/value detail and, for byte-range findings, a small
hex preview. A side panel: five check indicators (S1 pickle code, S2 unaccounted
bytes, S3 precision, S4 bit statistics, S5 base-model diff) each showing whether it
ran and what it found, plus a short plain-language legend of what each check does.
Buttons: "Export report (JSON)" and "Scan another".

Keep the same typography, cyan accent, hairline rules and dark background as the
main page. On phones the fact tiles and the findings/side-panel stack vertically.

---

## After Stitch

Export and upload the HTML. Then: the landing page becomes the site's front page,
the existing working scanner becomes the "Scan" page, the real figures are wired in
from the findings, and the whole `web/` build is deployed as a static site (the
`connect-src 'none'` CSP stays intact, so nothing is ever uploaded).
