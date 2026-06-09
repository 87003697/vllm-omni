#!/bin/bash
# FlowEdit image editing curl example (2 images required: source + condition)
#
# Usage:
#   bash run_curl_flowedit.sh source.png condition.png "Edit prompt"
#   bash run_curl_flowedit.sh input.png input.png "Make it snowy"  # single-image editing

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <source_image> <condition_image> \"<edit_prompt>\" [output_file]" >&2
  echo "" >&2
  echo "  source_image:    image being edited (image[0])" >&2
  echo "  condition_image: provides context (image[1])" >&2
  echo "  edit_prompt:     text prompt for editing" >&2
  echo "" >&2
  echo "For single-image editing, pass the same image twice." >&2
  exit 1
fi

SOURCE_IMG=$1
COND_IMG=$2
PROMPT=$3
SERVER="${SERVER:-http://localhost:8092}"
CURRENT_TIME=$(date +%Y%m%d%H%M%S)
OUTPUT="${4:-flowedit_${CURRENT_TIME}.png}"

# FlowEdit parameters (override via env)
CFG_SCALE_TGT="${CFG_SCALE_TGT:-7.5}"
CFG_SCALE_SRC="${CFG_SCALE_SRC:--7.5}"
N_MAX="${N_MAX:-28}"
NUM_STEPS="${NUM_STEPS:-28}"
SEED="${SEED:-42}"

for img in "$SOURCE_IMG" "$COND_IMG"; do
  if [[ ! -f "$img" ]]; then
    echo "Image not found: $img" >&2
    exit 1
  fi
done

REQUEST_JSON_FILE=$(mktemp)
trap 'rm -f "$REQUEST_JSON_FILE"' EXIT

# Encode both images and join with newline delimiter, then split in jq.
# This avoids the ARG_MAX limit that breaks `jq --arg` with large base64.
{
  base64 -w0 "$SOURCE_IMG" 2>/dev/null || base64 -i "$SOURCE_IMG"
  echo
  base64 -w0 "$COND_IMG" 2>/dev/null || base64 -i "$COND_IMG"
} | jq -Rs --arg prompt "$PROMPT" \
        --argjson cfg_tgt "$CFG_SCALE_TGT" \
        --argjson cfg_src "$CFG_SCALE_SRC" \
        --argjson n_max "$N_MAX" \
        --argjson steps "$NUM_STEPS" \
        --argjson seed "$SEED" \
  'split("\n") | {
    messages: [{
      role: "user",
      content: [
        {"type": "text", "text": $prompt},
        {"type": "image_url", "image_url": {"url": ("data:image/png;base64," + .[0])}},
        {"type": "image_url", "image_url": {"url": ("data:image/png;base64," + .[1])}}
      ]
    }],
    extra_body: {
      num_inference_steps: $steps,
      guidance_scale: 1,
      true_cfg_scale: $cfg_tgt,
      true_cfg_scale_src: $cfg_src,
      n_max: $n_max,
      seed: $seed
    }
  }' > "$REQUEST_JSON_FILE"

echo "FlowEdit image editing..."
echo "Server:         $SERVER"
echo "Prompt:         $PROMPT"
echo "Source image:   $SOURCE_IMG"
echo "Condition image: $COND_IMG"
echo "CFG scales:     tgt=$CFG_SCALE_TGT, src=$CFG_SCALE_SRC"
echo "n_max:          $N_MAX"
echo "Steps:          $NUM_STEPS"
echo "Output:         $OUTPUT"

curl -s "$SERVER/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @"$REQUEST_JSON_FILE" \
  | jq -r '.choices[0].message.content[0].image_url.url' \
  | cut -d',' -f2 \
  | base64 -d > "$OUTPUT"

if [[ -f "$OUTPUT" && -s "$OUTPUT" ]]; then
  echo "Image saved to: $OUTPUT"
  echo "Size: $(du -h "$OUTPUT" | cut -f1)"
else
  echo "Failed to generate image"
  exit 1
fi
