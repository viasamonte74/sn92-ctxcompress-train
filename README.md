# ctx-compress-slm

A specialised `ibm-granite/granite-4.0-h-micro` for the Microtensor `ctx_compress` arena (testnet 576,
arena v2, live since 1 Nov 2026), with everything needed to build, check and compare it: data from
the arena's own session generator, grader-checked training targets, long-context fine tuning,
optional pruning and 4-bit quantisation, evaluation the way the SLM pool scores, and packaging that
runs the validators' own checks.

Nothing here uploads to Hugging Face, commits on chain or rents machines.

## The arena in one table (live definition, 1 Nov 2026)

| | |
|---|---|
| Base (and reference) | `ibm-granite/granite-4.0-h-micro` @ `d5f01a3e`, 3.19B, 40 layers (Mamba-2, attention at 5/15/25/35), Apache 2.0 |
| Input | system: the compaction instruction with the token budget; user: the session rendered canonically in `<session>` tags, up to 122,880 tokens |
| Prompt | raw completion of `<\|start_of_role\|>system<\|end_of_role\|>{system}<\|end_of_text\|>\n<\|start_of_role\|>user<\|end_of_role\|>{user}<\|end_of_text\|>\n<\|start_of_role\|>assistant<\|end_of_role\|>`, greedy, max 3,072 tokens, stop `<\|end_of_text\|>` |
| Output | markdown with nine `## ` headings in order, nothing before the first: Primary request and intent, Key technical concepts, Files and code sections, Errors and fixes, Problem solving, All user messages, Pending tasks, Current work, Next step |
| Length gate | output tokens (Granite tokenizer) ≤ min(3072, max(1000, 5% of input)), else the item scores 0 |
| Item score | (1 − invented identifiers share) × (½ key-fact recall + ½ probe score); pass at ≥ 0.60 |
| Key facts | up to 12 files read/created/edited (edited first), the 5 latest exception types, 8 latest function names from edits and tracebacks |
| Probes | a Qwen2.5-1.5B-Instruct q8 GGUF on CPU reads only the summary: last edited file, changed a file yes/no, latest test pass/fail, created file, latest exception, function where the last traceback ended, next action (kind + file), which of four tasks |
| SLM score | Wilson LB (z 1.96) of the pass rate on 100 hidden items × twin factor (gap to half-cut twins, 0.02 tolerance, 0.1 slope) × (1 + 0.25·E); paid only if the paired LB against the base, clustered by family, is > 0; top five get 40/25/15/12/8% of the 20% SLM pool |
| Efficiency E | 0.4·TTFT + 0.3·memory + 0.2·parameters + 0.1·TPOT gains against the base, each clipped to [0, 1]; memory is computed: parameters × bytes per weight (bf16 2, GPTQ/AWQ 0.5) + 2·layers·8·64·125,952·2 bytes |
| Envelope (mt-5) | RTX 4090 (profile P1), vLLM 0.32 in batch invariant mode; TTFT p95 ≤ 4 s up to 32,768 input tokens, ≤ 25 s above with end to end ≤ 75 s; TPOT p95 ≤ 20 ms; at most 1% of items over |
| Weights | safetensors bf16/fp16 or 4-bit GPTQ/AWQ (legacy layout); fingerprint of layers 0–7 attention (here layer 5 only) ≥ 0.9 against the base; not byte identical to a public checkpoint |
| Evaluation | miners run every pool themselves on their own RTX 4090 in the pinned images and upload a signed transcript; validators replay on CPU and re-run ≥ 40 sampled calls on a 4090 with an exact token match |
| Epoch 0 | genesis block 8,348,450 (≈ 1 Nov 09:00 UTC); commit window 14,400 blocks (≈ 48 h, closes ≈ 3 Nov 09:00 UTC); epoch 50,400 blocks (≈ 7 days); fee 0 |

The hidden items come from the same generator `mta dev` uses (`microtensor-arenas/arenas/compress/synth.py`:
a coined Python project with one injected bug, worked by a scripted agent, real tool output) under a
secret seed, with sessions of 35k–95k tokens, full items ≥ 40k tokens and twins at the half cut.

## Strategy

1. **Data that matches the hidden pool.** `ccslm.data` drives the arena's own generator, cut rules
   (`build.choose_cuts`, `build.make_item`) and label code (`observe.labels`) with private seeds.
   Train sessions use the dev lengths (12k–95k) and add two extra random cuts per session (sessions
   are expensive, cuts are free); val and test copy the hidden shape. Each item carries an `oracle`
   block: what the session shows up to the cut, restricted to text visible in the rendering.
2. **Targets the grader cannot fault.** `ccslm.targets` writes the nine-heading summary from the
   oracle block: every label fact listed (recall 1.0), only identifiers that occur in the session
   (0 invented), the probe answers as plain `name: value` lines, the next action as
   `Next action: <kind> <file>` (file only if the session has shown it), trimmed to ≤ 85% of the
   budget (cap 1,400 tokens). Every target is re-checked with the grader's own functions; rejects are
   dropped. With the real probe model, the targets pass every item tried so far (see Results).
3. **Fine tune the base on the full-length prompts.** `ccslm.train` keeps only the completion
   logits (`logits_to_keep`), so 100k-token examples fit; LoRA r=64 on attention, Mamba `in_proj` and
   the shared MLP (the Mamba `out_proj` is fused by the training kernel, so an adapter there would
   never train); `--method full` for a full fine tune. The model learns to read a long session and
   emit the facts, which is the skill the hidden pool tests.
4. **Measure like the pool.** `ccslm.generate` calls a vLLM server exactly like the SLM pool (raw
   completion, greedy, stop token, streamed timings); `ccslm.grade` runs the validator's
   `compress.grade` with the Granite token counter and the probe, then reports the pass rate, Wilson
   LB, twin factor and the paired LB against the base.
5. **Small, cheap and fast, by the arena's own numbers** (`python -m ccslm.estimate plan`):
   - *Price class.* The system and escalation pools price every SLM token by parameter count
     (`r_class`: ≤ 1B 0.02, ≤ 3B 0.05, above 0.1 micro-dollars per token). Granite H Micro is
     3,191,396,096 parameters, so it pays the top rate, the same per input token as Claude Haiku 5.5
     and twice GLM 5.3 Flash. Cutting the 3 least useful Mamba layers (76.2M each) gives 2.963B and
     halves the price: variant C.
   - *SLM pool efficiency.* E against the base: dropping 3 layers ≈ +1.7% on the score, GPTQ ≈ +3%,
     10 layers + GPTQ ≈ +8%. Each 0.01 of lower bound lost costs ≈ 1.2%, so compression stays a
     variant that must beat A on the estimated score, not a default.
   - *Shorter answers.* The compact target style (default) keeps every fact and probe line in 31% fewer
     tokens (mean 1,033 → 717 on test, longest 1,391 → 977): about a third less decode time per request
     and more room under the 1,000-token minimum budget.
   - *The ≤ 1B class* (5× cheaper than Haiku) is out of reach for this base by pruning alone.
6. **Ship what the validators accept.** `ccslm.package` builds `mt_slm_ctx_compress` (weights, base
   tokenizer bytes, the base's own `config.json`, card under the Microtensor submission licence,
   training record) and runs preflight (`mt check-model`), `check_slm_repo`, the licence check, the
   fingerprint and the public-checkpoint check.

## Layout

```
ccslm/common.py     paths, live arena definition and drift check, Granite tokenizer, portable config, stage markers
ccslm/data.py       items in the arena's format from its generator, with oracle blocks (train / val / test)
ccslm/targets.py    nine-heading targets, budget trimming, grader checks, {prompt, completion} pairs
ccslm/probe.py      the probe agent: validator prompt; llama.cpp/OpenAI server or transformers
ccslm/grade.py      SLM pool grading and statistics: pass rate, Wilson LB, twin factor, paired LB, S_slm estimate
ccslm/generate.py   vLLM client like the SLM pool, with TTFT/TPOT; the envelope rule
ccslm/train.py      long-context SFT (LoRA, or full with fp32 master weights and 8-bit moments), numbered checkpoints and resume, merge
ccslm/export.py     merge any numbered checkpoint into a servable model (`--list`, `--step N`)
ccslm/prune.py      block-influence scores, Mamba layer cut (never layers 0-7 or attention), fingerprint report
ccslm/quantize.py   GPTQ 4-bit (experimental) and the validators' quantised-format rules
ccslm/package.py    slm_repo + validator checks + efficiency inputs
ccslm/estimate.py   parameters, price class, computed memory, E and cost against frontier models for pruning/quant plans
ccslm/pipeline.py   stages with logs and done markers; variants A (LoRA), B (A+GPTQ), C (under 3.0B + LoRA), D (C+GPTQ), F (full), G (8 layers cut + full)
scripts/setup_gpu.sh   GPU instance setup (vLLM 0.32, kernels, models, tests)
scripts/serve_vllm.sh  vLLM like the engine image (batch invariant, seed 0, 125,952 context)
tests/              targets, formats, envelope, streaming client, tiny-model train/prune/package, pipeline dry run
RUNBOOK.md          the GPU runbook, step by step with times and costs
```

## Quick start

On this CPU box (done for the default sizes; outputs in `out/data`):

```
. .venv/bin/activate
python -m ccslm.pipeline --stages check,data,targets
python -m pytest -q tests
```

Then follow RUNBOOK.md on a GPU instance.

## Results so far (CPU only, 1 Nov)

- Data (`out/data`, 0 label problems): train 995 sessions → 3,935 items (full, twin and two extra cuts; 179M prompt
  tokens, p50 41k, max 122k); val 58 sessions → 116 items; test 96 sessions → 192 items, hidden-shaped (full items
  ≥ 40k tokens, twins at the half cut), like one SLM pool.
- Targets: 3,935 + 116 + 192 pairs, 0 rejected: every one has valid headings, fits the budget, recall 1.0 and 0
  invented identifiers. Completion lengths 685–1,391 tokens on test.
- Probe model (transformers fp32 on CPU): 4/4 sample items pass with item scores 0.88–1.0. Rewording the Next step
  line lifted next-step probe accuracy from 0.46 to 0.83 on 12 items; the task-choice probe stays at 0.92. Oracle grade on 30
  hidden-shaped test sessions + their 30 twins (`out/eval/oracle-cpu/`): 60/60 pass, Wilson LB 0.887 (the most 30
  items allow), twin factor 1.0, mean item score 0.97, continuation 0.94 (choice 0.95, exact 0.96, next step 0.89).
  Known weak spot: when a later user message names another exception type, the probe sometimes answers that one.
- The real 3.19B Granite runs through `ccslm.train`: one LoRA step on CPU (torch fallback kernels), loss 1.94, finite
  gradient norm, 10 GB peak memory. LoRA hits 132 modules (40 + 40 shared MLP, 36 Mamba in_proj, 16 attention).
- Tests: 12 pass (`python -m pytest -q tests`).
- Not measured yet (needs a GPU): the base model's pass rate, the trained model, speed and the envelope.

## Gotchas found while building

- The Mamba training kernels must be real packages: transformers uses hub `kernels` only when asked, so with just
  `kernels` installed it falls back to the reference PyTorch scan, which runs out of memory at 16k tokens.
  `setup_gpu.sh` builds `causal-conv1d` and `mamba-ssm` from source with `--no-deps` (mamba-ssm would downgrade vLLM's
  `tilelang`/`apache-tvm-ffi` pins).
- transformers 5 saves Granite's config with renamed layer types (`linear_attention`/`full_attention`),
  `dtype` instead of `torch_dtype` and `time_step_limit: [0.0, {"__float__": "Infinity"}]`. Every exported model
  gets the base's own `config.json` back (`common.portable_config`); `package` refuses any other difference.
- In training mode with the Mamba kernels the layer fuses `out_proj` and reads its raw weight, so LoRA there is
  silently untrained; it is excluded from the targets.
- The next action's file is often one the agent has not opened yet; naming it counts as an invented identifier.
- The next-step probe's kind list has `search`, the grader's does not: a `search` next action can score at most 0.5.
- Plain AdamW on bf16 weights loses the updates of a full fine tune (a 1e-4 step is below bf16 resolution);
  `--method full` steps fp32 master copies. LoRA is unaffected: PEFT keeps adapters in fp32.
- The miner template still ships the v1 JSON adapter and Qwen chat markers; submit the SLM alone (the manifest's
  components list) until the harness is rewritten for the nine-heading markdown.
