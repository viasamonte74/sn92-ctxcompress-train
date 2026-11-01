#!/usr/bin/env bash
# Build and upload one checkpoint as a ctx_compress submission (slm component only), and print the
# chain commit for it. Nothing is committed on chain here.
#
#   scripts/submit_checkpoint.sh STEP [HF_USER]          e.g. scripts/submit_checkpoint.sh 230 Samonte
#
# 1. export + package checkpoint STEP into out/slm_repo/A-sNNNNN (the validators' checks must pass)
# 2. upload it as a new revision of <HF_USER>/mt_slm_ctx_compress (private) -> slm sha
# 3. build out/artifact/A-sNNNNN from the miner template, write the manifest (components slm, profile P1),
#    run `mt submit validate`, upload it as a new revision of <HF_USER>/microtensor_ctx_compress -> artifact sha
#
# The validators derive both repository names from the hf user, so they must be exactly these names.
# Several hotkeys can share the repositories: each commits its own artifact sha, which pins its own slm sha.
set -euo pipefail
STEP=${1:?checkpoint step}
HF_USER=${2:-Samonte}
ARENA=ctx_compress
PROFILE=P1
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
MT="$ROOT/.venv/bin/mt"
TEMPLATE=${MT_TEMPLATE:-$(cd "$ROOT/../../../microtensor-miner-template" 2>/dev/null && pwd || true)}
TAG=$(printf "A-s%05d" "$STEP")
SLM="$ROOT/out/slm_repo/$TAG"
ART="$ROOT/out/artifact/$TAG"
SLM_REPO="$HF_USER/mt_slm_$ARENA"
ART_REPO="$HF_USER/microtensor_$ARENA"
export HF_HOME=${HF_HOME:-/workspace/hf}
: "${HF_TOKEN:?export HF_TOKEN (write access to $HF_USER)}"
[[ -d "$TEMPLATE" ]] || { echo "miner template not found; set MT_TEMPLATE" >&2; exit 1; }
cd "$ROOT"

# 1. merged model + slm repository, checked the way the mirror checks it
[[ -f "out/runs/A/merged-step-$(printf %05d "$STEP")/config.json" ]] || \
  "$PY" -m ccslm.pipeline --stages export --variant A --step "$STEP"
"$PY" -m ccslm.pipeline --stages package --variant A --step "$STEP"
"$PY" - "$SLM" <<'EOF'
import json, sys
found = json.load(open(sys.argv[1] + ".checks.json"))
if not found["ok"]:
    sys.exit(f"slm checks failed: {found['problems']}")
print("slm checks ok, weights", found["preflight"]["weights_sha256"])
EOF
"$MT" check-model --slm "$SLM" --profile "$PROFILE" --arena "$ARENA"

# 2. the slm revision (delete_patterns drops files of an earlier checkpoint that this one lacks)
SLM_SHA=$("$PY" - "$SLM_REPO" "$SLM" "$TAG" <<'EOF'
import sys
from huggingface_hub import HfApi
repo, folder, tag = sys.argv[1:]
api = HfApi()
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
info = api.upload_folder(repo_id=repo, folder_path=folder, commit_message=f"slm {tag}", delete_patterns="*")
print(info.oid)
EOF
)
echo "slm        $SLM_REPO@$SLM_SHA"

# 3. the artifact: the template's runtime files (not executed: only the slm is submitted, the arena's
#    reference harness, router and escalation run around it) and the manifest pinning this slm revision
rm -rf "$ART"; mkdir -p "$ART"
for name in config.json main.py requirements.txt LICENSE README.md harness escalation; do
  [[ -e "$TEMPLATE/$name" ]] && cp -r "$TEMPLATE/$name" "$ART/"
done
find "$ART" -name __pycache__ -prune -exec rm -rf {} +
"$MT" submit manifest --artifact "$ART" --arena "$ARENA" --components slm \
  --slm "$SLM" --slm-repo "$SLM_REPO" --slm-revision "$SLM_SHA" --profile "$PROFILE"
"$MT" submit validate --artifact "$ART" --slm "$SLM" --allowed-bases ibm-granite/granite-4.0-h-micro
ART_SHA=$("$PY" - "$ART_REPO" "$ART" "$TAG" <<'EOF'
import sys
from huggingface_hub import HfApi
repo, folder, tag = sys.argv[1:]
api = HfApi()
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
info = api.upload_folder(repo_id=repo, folder_path=folder, commit_message=f"artifact {tag}", delete_patterns="*")
print(info.oid)
EOF
)
echo "artifact   $ART_REPO@$ART_SHA"
printf '{"step": %d, "slm_repo": "%s", "slm_sha": "%s", "artifact_repo": "%s", "artifact_sha": "%s"}\n' \
  "$STEP" "$SLM_REPO" "$SLM_SHA" "$ART_REPO" "$ART_SHA" | tee -a "$ROOT/out/submissions.jsonl"
echo
echo "commit with:  mt submit commit --arena $ARENA --hf-user $HF_USER --artifact-sha $ART_SHA --profile $PROFILE \\"
echo "                --network test --netuid 576 --wallet-name rabi --wallet-hotkey <hotkey> --home ~/.microtensor/<hotkey> \\"
echo "                --fee-block <block> --fee-index <index>"
