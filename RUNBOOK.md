# GPU runbook

Times marked ~ are estimates: measure them on the first run and correct this file. Prices are rough
vast.ai figures; check the offer.

## 0. What to rent

| Job | GPU | Why | Disk |
|---|---|---|---|
| training | 1 × H100 80 GB (A100 80 GB works, ~1.5–2× slower) | 100k-token examples need the memory; bf16 | 200 GB |
| baseline, eval | the training box is fine | quality does not depend on the card | |
| speed, envelope, the miner run | 1 × RTX 4090 24 GB | profile P1: the envelope and the spot checks are on a 4090 | 100 GB |

Image: any CUDA 12.8 image with Python 3.10+ and git (for example a vast.ai PyTorch image).
H100 ≈ $1.8–2.5/h, RTX 4090 ≈ $0.3–0.5/h.

## 1. Copy the kit (from this box)

The zip holds the code and the three pinned repositories, no data:

```
scp -P <port> /home/justin/work120/microtensor/ctx-compress-slm.zip root@<host>:/workspace/
ssh -p <port> root@<host> 'cd /workspace && unzip -q ctx-compress-slm.zip'
```

## 2. Set up and build the data (~45 min)

```
cd /workspace/microtensor/ctx-compress-slm
export HF_HOME=/workspace/hf
bash scripts/setup_gpu.sh            # vLLM 0.32, kernels, models, then the test suite
. .venv/bin/activate
python -m ccslm.pipeline --stages check            # stops if the live arena changed
python -m ccslm.pipeline --stages data,targets     # CPU: ~20 min on 8 cores, faster with more
```

The data comes from the arena's generator under fixed seeds (ccslm-train, ccslm-val, ccslm-test), so
every box builds the same projects, bugs and cuts; only details of the real tool output (pytest
timings) differ. About 4,000 train, 116 val and 192 test items, 1.1 GB. The generator runs pytest,
which setup_gpu.sh installs.
To skip the build, copy `out/data` from a box that has it instead
(`rsync -a out/data root@<host>:/workspace/microtensor/ctx-compress-slm/out/`).

## 3. Reference numbers (~45 min)

```
python -m ccslm.pipeline --stages oracle      # ceiling of the target format with the probe (~10 min)
python -m ccslm.pipeline --stages baseline    # untrained base on the 192 test items (~30 min)
cat out/eval/oracle/report.json out/eval/base/report.json
```

For the validators' exact probe instead of transformers, run llama.cpp with the pinned GGUF and point
the pipeline at it:

```
docker run -d -p 8090:8000 -v $HF_HOME:/hf ghcr.io/ggml-org/llama.cpp:server \
  -m /hf/<path to qwen2.5-1.5b-instruct-q8_0.gguf> --host 0.0.0.0 --port 8000 -c 8192 -np 1 -t 8
export CCSLM_PROBE_URL=http://127.0.0.1:8090
```

## 4. Train variant A (LoRA r=64)

First a short run to see that it learns and to measure the speed:

```
CCSLM_TRAIN_ARGS="--max-steps 40" python -m ccslm.pipeline --stages train --variant A
tail -f out/logs/train-A.log        # one json line per optimizer step: loss, sequence tokens, seconds, memory
```

40 steps × 16 examples ≈ 640 examples ≈ 30M tokens. Read `elapsed_s` per step in
`out/runs/A/metrics.jsonl` and scale: one epoch of ~4,000 pairs is ~250 steps. If the loss is
already flat by step 40, evaluate before paying for more. To go on, clear the stage marker and run
again; the checkpoint resumes:

```
rm out/stages/train-A.done
CCSLM_TRAIN_ARGS="--max-steps 250" python -m ccslm.pipeline --stages train --variant A
```

Out of memory: add `--max-tokens 64000` to `CCSLM_TRAIN_ARGS` (drops the longest examples).
An interruption is safe: rerun the same command and it resumes from `out/runs/A/checkpoint`.

## 5. Evaluate and package A (~40 min)

```
python -m ccslm.pipeline --stages eval,package --variant A
cat out/eval/A/report.json          # pass rate, LB, twin factor, paired LB against the base
cat out/slm_repo/A.checks.json      # validator checks: must say "ok": true
```

Copy the results back after every stage:

```
rsync -a root@<host>:/workspace/ctx-compress-slm/out/{eval,stages,logs,slm_repo} ./out/ -e 'ssh -p <port>'
```

## 6. Speed and the envelope (on the RTX 4090)

Copy `out/runs/A/merged` (or `out/slm_repo/A`) and `out/data/test` to the 4090 box, run the setup,
then:

```
python -m ccslm.pipeline --stages speed,report --variant A
```

That times the base and A at concurrency 1 on the 30 longest test items and writes
`out/speed/*.envelope.json`; `report` prints the variants table with the estimated SLM score.

## 7. Variants (only if A is solid)

First look at what each would buy (CPU, seconds): `python -m ccslm.estimate plan`.

```
python -m ccslm.pipeline --stages prune,train,eval,speed,package,report --variant C    # under 3.0B params: half the token price
python -m ccslm.pipeline --stages quant,eval,speed,package,report --variant D          # C + GPTQ (needs setup_gpu.sh --quant)
python -m ccslm.pipeline --stages quant,eval,speed,package,report --variant B          # A + GPTQ
python -m ccslm.pipeline --stages train,eval,speed,package,report --variant F          # full fine tune, 80 GB
```

Keep a variant only if its estimated SLM score beats A's and its paired LB against the base stays > 0.
C is the one to try first: 3 layers of 40 is a mild cut, and it is what makes the model cheaper than
Claude Haiku in the system and escalation pools.

## 8. Stop the instances

After copying `out/` back, stop or destroy the instances; they bill until then.

## 9. Submitting (yours to do)

1. Push `out/slm_repo/<variant>` to `huggingface.co/<user>/mt_slm_ctx_compress`; note the commit sha.
2. Fork the miner template into `<user>/microtensor_ctx_compress`, submit the SLM only (its harness
   still targets the old JSON format), then:
   `mt submit manifest --artifact . --arena ctx_compress --slm ./slm_repo --slm-repo <user>/mt_slm_ctx_compress --slm-revision <sha> --profile P1`,
   `mt submit validate`, `mt check-model --slm ./slm_repo --profile P1 --arena ctx_compress`.
3. `mt submit commit --arena ctx_compress --hf-user <user> --artifact-sha <sha> --profile P1` inside the
   commit window (epoch 0 closes ≈ 3 Nov 09:00 UTC; epoch 1's window ≈ 8 Nov 09:00 → 10 Nov 09:00 UTC).
4. During the evaluation window, run the official runner on an RTX 4090
   (`microtensor-subnet/deploy/docker-compose.miner.yml`) and keep the upload receipt.

On this box use the venv's `mt` (or `python -m microtensor_subnet`): `/usr/bin/mt` is GNU cpio's tape tool.
