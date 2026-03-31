  SPK=${1:-spk1}
  VID=${2:-both}  # vid1, vid2, or both
  GPUS=${3:-"0,1"}  # comma-separated GPU indices, e.g. "0,1" or "2,3" or "0"
  TEST_MODE=false  # Set to true to process only the first video

  # Parse GPU indices into array
  IFS=',' read -r -a GPU_IDS <<< "$GPUS"
  NUM_GPUS=${#GPU_IDS[@]}

  INPUT_DIR="/data1/span_data/prompt/data/assessments/$SPK/video/split"
  OUTPUT_DIR="/data1/span_data/prompt/data/assessments/$SPK/video/split_blur"
  FACE_MODEL="/data1/span_data/prompt/src/EgoBlur/models/ego_blur_face_gen2.jit"

  mkdir -p "$OUTPUT_DIR"

  # Collect eligible files
  files=()
  for f in "$INPUT_DIR"/*.mov; do
    filename=$(basename "$f")
    if [[ "$filename" == *vmpac_r* ]]; then
      echo "Skipping: $filename"
      continue
    fi
    base="${filename%.*}"
    if [[ "$VID" == "vid1" && "$base" != *_vid1 ]]; then
      continue
    elif [[ "$VID" == "vid2" && "$base" != *_vid2 ]]; then
      continue
    fi
    files+=("$f")
    if [ "$TEST_MODE" = true ]; then
      break
    fi
  done

  echo "Processing ${#files[@]} files across $NUM_GPUS GPUs"

  # Launch videos round-robin across GPUs
  pids=()
  for i in "${!files[@]}"; do
    f="${files[$i]}"
    gpu_id="${GPU_IDS[$((i % NUM_GPUS))]}"
    filename=$(basename "$f")
    echo "Dispatching: $filename -> GPU $gpu_id"
    CUDA_VISIBLE_DEVICES=$gpu_id egoblur-gen2 \
      --face_model_path "$FACE_MODEL" \
      --input_video_path "$f" \
      --output_video_path "$OUTPUT_DIR/$filename" \
      --num_gpus 1 &
    pids+=($!)

    # If we've filled all GPU slots, wait for one to finish before continuing
    if (( ${#pids[@]} >= NUM_GPUS )); then
      wait "${pids[0]}"
      echo "Done: $(basename "${files[$((i - NUM_GPUS + 1))]}")"
      pids=("${pids[@]:1}")
    fi
  done

  # Wait for remaining jobs
  for pid in "${pids[@]}"; do
    wait "$pid"
  done

  echo "All files processed."

  #usage: bash egoblur_run.sh spk# [vid1|vid2|both] [gpu_ids]
  #  e.g.: bash egoblur_run.sh spk1 both 0,1
  #        bash egoblur_run.sh spk2 vid2 2,3