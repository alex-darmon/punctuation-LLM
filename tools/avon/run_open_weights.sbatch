#!/bin/bash
# One sitting of the open_weights_protocol_v1 campaign: serve the pinned
# weights, generate both arms against them, stop the server.
#
#   sbatch tools/avon/run_open_weights.sbatch smoke
#   sbatch tools/avon/run_open_weights.sbatch full [--resume]
#
# The server and the client are deliberately different environments.  The
# client's spaCy is the stopping rule (count_marks), so it is pinned to the
# repository's requirements and kept out of reach of vLLM's resolver.
#
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH -c 24
#SBATCH --mem=120G
#SBATCH -t 2-00:00:00
#SBATCH -o logs/%x_%j.log

set -uo pipefail

MODE="${1:?smoke|full}"; shift || true
R="$HOME/punct-open-arm"
REPO="$R/repo"
GEN_PY="$R/envs/gen/bin/python"
VLLM_PY="$R/envs/vllm/bin/python"
CONFIG="campaigns/open_weights_protocol_v1.json"

export HF_HUB_OFFLINE=1          # weights are already local and revision-pinned
export PYTHONNOUSERSITE=1        # never let ~/.local shadow either environment
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset PYTHONPATH                 # a leaked PYTHONPATH defeats PYTHONNOUSERSITE

cd "$REPO" || exit 1

# Preflight.  Both environments are checked before the GPUs are touched: a
# missing import discovered after a 10-minute model load costs an allocation,
# and an environment that silently resolves a package from outside its own
# prefix is how the mark counter changes without the protocol changing.
preflight() {
    "$GEN_PY" - <<'PY' || return 1
import sys
sys.path.insert(0, ".")
import generate_positional_mechanism as gpm
env = gpm.client_environment()
outside = env["modules_outside_prefix"]
if outside:
    raise SystemExit(f"client env resolves {outside} outside {sys.prefix}")
if gpm.legacy_protocol_sha256() != gpm.LEGACY_PROTOCOL_SHA256:
    raise SystemExit("prompt templates no longer match the frozen protocol hash")
if gpm.count_marks("Hello, world. Is it? Yes; truly: it is!") != 6:
    raise SystemExit("mark counter disagrees with the reference count")
print(f"[preflight] client ok: spacy {env['spacy']['version']}, "
      f"en_core_web_sm {env['en_core_web_sm']['version']}")
PY
    "$VLLM_PY" -c "import vllm, torch; print(f'[preflight] server ok: vllm {vllm.__version__}, torch {torch.__version__}')" || return 1
    "$VLLM_PY" -m pip check >/dev/null 2>&1 || { echo "[preflight] server dependency set is inconsistent"; return 1; }
}
preflight || { echo "[run] FATAL: preflight failed; not allocating the model"; exit 1; }

MODEL_NAME=$("$GEN_PY" -c "import json;print(json.load(open('$CONFIG'))['generation']['model']['name'])")
REVISION=$("$GEN_PY"  -c "import json;print(json.load(open('$CONFIG'))['generation']['model']['revision'])")
TP=$("$GEN_PY"        -c "import json;print(json.load(open('$CONFIG'))['generation']['model']['tensor_parallel_size'])")
DTYPE=$("$GEN_PY"     -c "import json;print(json.load(open('$CONFIG'))['generation']['model']['dtype'])")

# Serve the revision's snapshot directory, not the repo id: that is what makes
# the pin real rather than a label in a config file.
SNAPSHOT=$("$VLLM_PY" -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$MODEL_NAME', revision='$REVISION', local_files_only=True))
" 2>/tmp/snap_err.$$) || {
    echo "[run] FATAL: could not resolve the pinned revision locally:"
    sed 's/^/    /' "/tmp/snap_err.$$"
    rm -f "/tmp/snap_err.$$"
    exit 1
}
rm -f "/tmp/snap_err.$$"

# WORKERS is the campaign's declared worker count: the number of runs in
# flight, hence the batch composition the manifest records.  Measured against
# this model's KV-cache headroom by the smoke test, not inherited from anywhere.
WORKERS="${OPEN_WEIGHTS_WORKERS:-8}"

PORT=$(( 20000 + SLURM_JOB_ID % 10000 ))
echo "[run] mode=$MODE model=$MODEL_NAME rev=${REVISION:0:12} tp=$TP dtype=$DTYPE port=$PORT workers=$WORKERS"
echo "[run] snapshot=$SNAPSHOT"
nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version --format=csv

"$VLLM_PY" -m vllm.entrypoints.openai.api_server \
    --model "$SNAPSHOT" \
    --served-model-name "$MODEL_NAME" \
    --tokenizer "$SNAPSHOT" \
    --dtype "$DTYPE" \
    --tensor-parallel-size "$TP" \
    `# 16384 was sized from the excerpt (at most 2,996 tokens over all 20` \
    `# authors) plus the 8192-token cap.  A run in job 2210361 nonetheless` \
    `# sent 8,410 prompt tokens and was rejected (16,602 > 16384).  Typical` \
    `# calls run 1.3 tokens per word, so the excess is not ordinary prose;` \
    `# which run produced it was lost because the smoke runner had no` \
    `# checkpointing then.  The cap is set from the observed maximum rather` \
    `# than from a model of where the tokens went.` \
    `# 32768 fixed that but is not free - vLLM profiles activation peak at the` \
    `# declared length, which grew 1.54 -> 2.84 GiB and cut KV cache from 4.78` \
    `# to 3.49 GiB per GPU.  20480 clears the worst observed request with` \
    `# margin while keeping the KV cache that concurrency depends on.` \
    --max-model-len 20480 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs "$WORKERS" \
    --port "$PORT" \
    --disable-log-requests &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT INT TERM

for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 $SERVER 2>/dev/null || { echo "[run] FATAL: vLLM exited during startup"; exit 1; }
    sleep 10
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "[run] FATAL: server never became healthy"; exit 1; }
echo "[run] server healthy after $SECONDS s; version: $(curl -s http://127.0.0.1:$PORT/version)"

export VLLM_BASE_URL="http://127.0.0.1:$PORT/v1"
export SOURCE_COMMIT="$(cat "$REPO/.source_commit" 2>/dev/null)"
export SLURM_NODE_GPUS="$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"

if [ "$MODE" = "smoke" ]; then
    "$GEN_PY" tools/open_weights_smoke.py --config "$CONFIG" --parallel "$WORKERS" "$@"
    STATUS=$?
else
    "$GEN_PY" generate_positional_mechanism.py \
        --config "$CONFIG" --parallel "$WORKERS" "$@"
    STATUS=$?
fi
echo "[run] finished mode=$MODE status=$STATUS after $SECONDS s"
exit $STATUS
