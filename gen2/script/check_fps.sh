
#!/bin/bash

# --- Configure these paths ---
ORIG_DIR="/data1/span_data/prompt/data/assessments/spk1/video/split"
BLUR_DIR="/data1/span_data/prompt/data/assessments/spk1/video/split_blur"

# Print header
printf "%-60s %10s %10s\n" "Video" "Orig FPS" "Blur FPS"
printf "%-60s %10s %10s\n" "-----" "--------" "--------"

# Iterate over blurred videos
find "$BLUR_DIR" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" -o -name "*.mov" \) | sort | while read -r blur_file; do
    # Get relative path from blur dir
    rel_path="${blur_file#$BLUR_DIR/}"

    # Find counterpart in original dir
    orig_file="$ORIG_DIR/$rel_path"

    if [[ ! -f "$orig_file" ]]; then
        printf "%-60s %10s %10s\n" "$rel_path" "NOT FOUND" "$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$blur_file" 2>/dev/null | bc -l | xargs printf '%.2f')"
        continue
    fi

    # Get FPS for both
    orig_fps=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$orig_file" 2>/dev/null | bc -l | xargs printf '%.2f')
    blur_fps=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$blur_file" 2>/dev/null | bc -l | xargs printf '%.2f')

    # Flag mismatch
    if [[ "$orig_fps" != "$blur_fps" ]]; then
        flag=" *** MISMATCH ***"
    else
        flag=""
    fi

    printf "%-60s %10s %10s%s\n" "$rel_path" "$orig_fps" "$blur_fps" "$flag"
done