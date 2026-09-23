#!/bin/bash
# Move the open-weights arm between this checkout and Avon.
#
#   tools/avon/sync.sh push     code + campaigns + the source books it reads
#   tools/avon/sync.sh pull     generated text, call records and manifests
#
# Only the files the generator actually opens are sent.  The cluster copy is
# not a clone: no results/, no caches, no other campaigns' output.
set -euo pipefail

HOST="${AVON_HOST:-marmjx@avon.scrtp.warwick.ac.uk}"
REMOTE="${AVON_ROOT:-punct-open-arm}"
LOCAL="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="generated_texts_open_weights_v1"

case "${1:?push|pull}" in
push)
    # The commit is recorded in the run manifest, so refuse to push a dirty
    # tree silently: what generated the text must be identifiable afterwards.
    cd "$LOCAL"
    COMMIT="$(git rev-parse HEAD)"
    if ! git diff --quiet HEAD -- generate_positional_mechanism.py punctlib campaigns; then
        echo "warning: protocol files differ from HEAD; the manifest will record $COMMIT anyway" >&2
    fi
    # openrsync (the macOS rsync) does not create intermediate remote
    # directories, so every destination is made explicitly first.
    ssh "$HOST" "mkdir -p '$REMOTE/repo' '$REMOTE/logs' \
        '$REMOTE/repo/tools/avon' \
        '$REMOTE/repo/punctuation-stylometry-master/punctuation' \
        '$REMOTE/repo/punctuation-stylometry-master/conf' \
        '$REMOTE/repo/punctuation-stylometry-master/logs' \
        '$REMOTE/repo/full_books' '$REMOTE/repo/gutenberg_texts'"
    # macOS ships openrsync (protocol 29); --delete-excluded is not portable there.
    rsync -az generate_positional_mechanism.py "$HOST:$REMOTE/repo/"
    rsync -az --delete punctlib/ "$HOST:$REMOTE/repo/punctlib/"
    rsync -az --delete campaigns/ "$HOST:$REMOTE/repo/campaigns/"
    rsync -az --delete tools/avon/ "$HOST:$REMOTE/repo/tools/avon/"
    # Glob rather than a hand-kept list: open_weights_inspect.py was added to
    # tools/ after this script and was silently never pushed, so it looked
    # broken on the cluster while working fine locally.
    rsync -az tools/open_weights_*.py tools/expand_open_weights_campaign.py \
        "$HOST:$REMOTE/repo/tools/"
    # The vendored parser: the package and its .ini only.  model/ and results/
    # are 60 MB of the original authors' artifacts that nothing here imports.
    rsync -az --delete \
        punctuation-stylometry-master/punctuation/ \
        "$HOST:$REMOTE/repo/punctuation-stylometry-master/punctuation/"
    rsync -az --delete \
        punctuation-stylometry-master/conf/ \
        "$HOST:$REMOTE/repo/punctuation-stylometry-master/conf/"
    # Only the books the campaign's authors declare.
    python3 - "$HOST" "$REMOTE" <<'PY'
import json, subprocess, sys, tempfile
host, remote = sys.argv[1], sys.argv[2]
cfg = json.load(open("campaigns/open_weights_protocol_v1.json"))
paths = sorted({bp for a in cfg["generation"]["authors"] for bp in a["book_paths"]})
with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
    fh.write("\n".join(paths) + "\n")
    listing = fh.name
subprocess.run(
    ["rsync", "-az", f"--files-from={listing}", ".", f"{host}:{remote}/repo/"], check=True
)
print(f"  {len(paths)} source books")
PY
    echo "$COMMIT" | ssh "$HOST" "cat > '$REMOTE/repo/.source_commit'"
    ssh "$HOST" "chmod +x '$REMOTE/repo/tools/avon/'*.sh 2>/dev/null; true"
    echo "pushed $COMMIT -> $HOST:$REMOTE/repo"
    ;;
pull)
    cd "$LOCAL"
    mkdir -p "$OUT_DIR"
    # No --info=progress2: macOS ships openrsync, which rejects it and prints
    # its usage instead of transferring anything.  The push path avoids the
    # same trap; this one did not, and a "pull" silently brought back nothing.
    mkdir -p "$OUT_DIR/_cluster_logs" "${OUT_DIR}_smoke"
    rsync -az "$HOST:$REMOTE/repo/$OUT_DIR/" "$OUT_DIR/" || true
    rsync -az "$HOST:$REMOTE/repo/${OUT_DIR}_smoke/" "${OUT_DIR}_smoke/" || true
    rsync -az "$HOST:$REMOTE/logs/" "$OUT_DIR/_cluster_logs/" || true
    echo "runs pulled: $(find "$OUT_DIR" -name 'run_*.txt' -not -path '*/raw/*' | wc -l | tr -d ' ')"
    ;;
*)
    echo "usage: $0 push|pull" >&2; exit 2 ;;
esac
