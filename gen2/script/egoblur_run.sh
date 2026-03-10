TEST_MODE=true  # Set to true to process only the first video

INPUT_DIR="/data1/span_data/prompt/data/assessments/spk1/video/split"
OUTPUT_DIR="/data1/span_data/prompt/data/assessments/spk1/video/split_blur"
FACE_MODEL="/data1/span_data/prompt/code/EgoBlur/models/ego_blur_face_gen2.jit"

mkdir -p "$OUTPUT_DIR"

for f in "$INPUT_DIR"/*.mov; do
  filename=$(basename "$f")
  if [[ "$filename" == *vmpac_r* ]]; then
    echo "Skipping: $filename"
    continue
  fi
  echo "Processing: $filename"
  egoblur-gen2 \
    --face_model_path "$FACE_MODEL" \
    --input_video_path "$f" \
    --output_video_path "$OUTPUT_DIR/$filename" \
    --num_gpus 2
  echo "Done: $filename"
  if [ "$TEST_MODE" = true ]; then
    echo "Test mode: stopping after first video."
    break
  fi
done

echo "All files processed."                                  

# egoblur-gen2 \
#     --face_model_path "/data1/span_data/prompt/code/EgoBlur/models/ego_blur_face_gen2.jit" \
#     --input_video_path "/data1/span_data/prompt/data/assessments/spk1/video/spk1_vid1.mov.nosync_100frames.mov" \
#     --output_video_path "/data1/span_data/prompt/data/assessments/spk1/video/spk1_vid1_blur.mov.nosync_100frames.mov" \
#     --num_gpus 2