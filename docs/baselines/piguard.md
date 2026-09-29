# PIGuard detector

`evaluate/defense_methods/piguard.py` loads the authors' released model through
`AutoModelForSequenceClassification` and returns the probability of injection.
It does not implement a replacement classifier, keyword filter, chunking scheme,
or fallback score. The surrounding retrieval and lifecycle policies are MAPLE
deployment variants; they are not changes to the PIGuard model.

## Pinned upstream artifacts

| Artifact | Pin |
| --- | --- |
| [Official source](https://github.com/leolee99/PIGuard/tree/1b5751e88bf7475acbedfc8eda795ce060307c84) | `1b5751e88bf7475acbedfc8eda795ce060307c84` |
| [Official HF model and tokenizer](https://huggingface.co/leolee99/PIGuard/tree/dd78b24e330193a22d2293ac66922dd4f982f563) | `leolee99/PIGuard`, revision `dd78b24e330193a22d2293ac66922dd4f982f563` |
| Official weight LFS metadata | SHA-256 `f90b9806de93b6286cda517300d4b55e5ce2e5ccbf8339dc59be21ca0dd9a25e`, 737,719,272 bytes |

The entire weight file was verified on inference1 on 2026-09-29: see [artifact evidence](evidence/piguard-artifacts.json). The wrapper was written independently;
it does not vendor upstream source. PIGuard source is [MIT licensed, copyright
(c) 2024 Hao Li](https://github.com/leolee99/PIGuard/blob/1b5751e88bf7475acbedfc8eda795ce060307c84/LICENSE).
Keep the upstream license with any future vendored code or model distribution.

## Selected inference profile

Both model and tokenizer receive the same revision, `trust_remote_code=True`,
and the tokenizer uses `use_fast=True`. Trusting the pinned custom code is
required: the official HF implementation uses the first token's hidden state
followed by a `Linear(768, 2)` classifier. A standard DeBERTa sequence classifier's
pooling head is not an interchangeable implementation.

The default follows the HF deployment `inference_examples.py`: truncation to
**512 tokens**, with batch padding, FP32 parameters/logits, `eval()`, and
`torch.inference_mode()`. The official GitHub `eval_hf.py` instead configures
2048 tokens, and the legacy GitHub `PIGuard.classify` uses 256. We select the HF
512-token deployment profile explicitly; this does not claim exact reproduction
of every reported paper evaluation. Long text is passed intact to the tokenizer
once and truncated there, so material beyond its retained tokens is not scored.
There is no sliding window or aggregation across chunks.

The official labels must be exactly `0: benign` and `1: injection` with two
classes. `score(text)` and `score_batch(texts)` return
`softmax(logits, dim=-1)[:, 1]`, **not the confidence of whichever label wins**.
At the default threshold of 0.5, block only when the returned score is greater
than 0.5. Equal logits produce 0.5 and are allowed, matching two-class argmax's
choice of class 0 on a tie. The runtime owns the threshold decision.

## API and provenance

```python
from types import SimpleNamespace
from evaluate.defense_methods.piguard import PIGuardDetector

detector = PIGuardDetector(SimpleNamespace())  # default pinned release, CPU
probability = detector.score("Text to inspect")
probabilities = detector.score_batch(["First text", "Second text"])
metadata = detector.provenance()
```

Supported args are `piguard_model`, `piguard_revision`, `piguard_device`
(default `cpu`), `piguard_max_length` (default `512`), and `piguard_threshold`
(default `0.5`, reported for runtime policy). Alternative model/revision/length
settings are recorded and must be disclosed when comparing results.

The JSON-compatible `provenance()` snapshot reports source/model/tokenizer pins,
label mapping, profile, input length, threshold, device, precision, and counters.
`calls` counts scoring requests, `model_calls` counts attempted model forwards,
`texts_scored` counts successfully returned scores, and `errors` counts failed
scoring requests. An empty batch returns an empty list without a model forward.
Injected loading functions are a test boundary and are disclosed in provenance;
their successful execution is not checkpoint validation.

Missing dependencies/checkpoints, incompatible labels, invalid input, malformed
logits, and non-finite logits fail explicitly. No failure is converted into an
allow/block probability.

## Validation and operational readiness

Run the deterministic inference contract tests from the repository root:

```bash
/mnt/public/data/wj/venvs/maple-baselines/bin/python -m unittest discover -s tests -p test_piguard.py -v
```

These tests use real PyTorch tensors with small deterministic logits and injected
external model/tokenizer loaders. They verify exact injection-probability
extraction, benign/injection/tie semantics, pinned loading arguments, FP32/eval/
inference mode, explicit 512-token truncation, strict labels, failures, and
provenance. They do not establish full-checkpoint numerical equivalence.

On 2026-09-29 the pinned snapshot was verified at
/mnt/public/data/wj/baseline-references/PIGuard-hf-dd78b24. Real offline CPU
FP32 inference matched the official HF pipeline on four inputs (benign,
injection, empty and long/truncated): maximum batch probability difference
8.94e-8, single-input difference 1.49e-8, with matching winning labels.
See [real checkpoint parity evidence](evidence/piguard-cpu-parity.json).
This is not benchmark performance or an equivalence claim for every input.

Pinned inference dependencies were installed only in the task venv:
Transformers4.44.0, tokenizers0.19.1, hub0.24.5, safetensors0.4.4,
sentencepiece0.2.0 and protobuf4.25.4. Existing Torch2.14.0+cpu differs
from upstream2.4.0; all installed-version deviations are in the evidence.
No shared model service was modified. Local-directory provenance reports
official_model_requested=false because the request is a local path;
the verified artifact hashes establish this snapshot's official identity.

The retrieval-only and lifecycle wrappers share the detector settings. The
lifecycle wrapper checks write, promotion, retrieval and transfer. That placement
is a MAPLE experiment, not an author-released PIGuard lifecycle algorithm.
Repeated prompt authorization reuses within-task read scores instead of making
duplicate inference calls. Measure clean utility/false positives, attack success,
calls, latency and effects of truncation before claiming superiority.
