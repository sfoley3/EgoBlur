#!/usr/bin/env bash
# quick_split.sh – Extract exactly 100 frames from a video.
# Usage: bash quick_split.sh <input_video> [output_video]

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <input_video> [output_video]"
  exit 1
fi

INPUT="$1"
OUTPUT="${2:-${INPUT%.*}_100frames.${INPUT##*.}}"

# Detect the frame rate of the source video
FPS=$(ffprobe -v error -select_streams v:0 \
  -show_entries stream=r_frame_rate \
  -of default=noprint_wrappers=1:nokey=1 "$INPUT")

# Convert fractional fps (e.g. 30000/1001) to a decimal
FPS_DEC=$(python3 -c "print(${FPS})")

# Duration in seconds for exactly 100 frames
DURATION=$(python3 -c "print(f'{100 / ${FPS_DEC}:.6f}')")

echo "Source FPS : $FPS ($FPS_DEC)"
echo "Duration   : ${DURATION}s (100 frames)"
echo "Output     : $OUTPUT"

ffmpeg -i "$INPUT" -t "$DURATION" -c copy "$OUTPUT"

echo "Done – wrote $OUTPUT"
