# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from gen2.script.constants import (
    FACE_THRESHOLDS_GEN2,
    LP_THRESHOLDS_GEN2,
    RESIZE_MAX_GEN2,
    RESIZE_MIN_GEN2,
)
from gen2.script.detectron2.export.torchscript_patch import patch_instances
from gen2.script.predictor import ClassID, EgoblurDetector, PATCH_INSTANCES_FIELDS
from gen2.script.utils import (
    get_all_devices,
    get_device,
    get_image_tensor,
    read_image,
    scale_box,
    setup_logger,
    validate_inputs,
    write_image,
)
from moviepy.editor import ImageSequenceClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm.auto import tqdm


logger = setup_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera_name",
        required=False,
        type=str,
        default=None,
        choices=[
            "slam-front-left",
            "slam-front-right",
            "slam-side-left",
            "slam-side-right",
            "camera-rgb",
        ],
        help=(
            "Optional camera identifier used to pick camera-specific default thresholds "
            "(see README for mapping details)."
        ),
    )
    parser.add_argument(
        "--face_model_path",
        required=False,
        type=str,
        default=None,
        help="Absolute EgoBlur Gen2 face model file path",
    )

    parser.add_argument(
        "--face_model_score_threshold",
        required=False,
        type=float,
        default=None,
        help="Face model score threshold to filter out low confidence detections",
    )

    parser.add_argument(
        "--lp_model_path",
        required=False,
        type=str,
        default=None,
        help="Absolute EgoBlur Gen2 license plate model file path",
    )

    parser.add_argument(
        "--lp_model_score_threshold",
        required=False,
        type=float,
        default=None,
        help="License plate model score threshold to filter out low confidence detections",
    )

    parser.add_argument(
        "--nms_iou_threshold",
        required=False,
        type=float,
        default=0.5,
        help="NMS iou threshold to filter out low confidence overlapping boxes",
    )

    parser.add_argument(
        "--scale_factor_detections",
        required=False,
        type=float,
        default=1,
        help="Scale detections by the given factor to allow blurring more area, 1.15 would mean 15%% scaling",
    )

    parser.add_argument(
        "--input_image_path",
        required=False,
        type=str,
        default=None,
        help="Absolute path for the given image on which we want to make detections",
    )

    parser.add_argument(
        "--output_image_path",
        required=False,
        type=str,
        default=None,
        help="Absolute path where we want to store the visualized image",
    )

    parser.add_argument(
        "--input_video_path",
        required=False,
        type=str,
        default=None,
        help="Absolute path for the given video on which we want to make detections",
    )

    parser.add_argument(
        "--output_video_path",
        required=False,
        type=str,
        default=None,
        help="Absolute path where we want to store the visualized video",
    )

    parser.add_argument(
        "--num_gpus",
        required=False,
        type=int,
        default=1,
        help="Number of GPUs to use for parallel frame processing (default: 1)",
    )

    return parser.parse_args()


def _get_threshold(
    camera_name: Optional[str],
    user_threshold: Optional[float],
    threshold_map: Optional[Dict[str, float]],
) -> float:
    """
    Resolve the effective score threshold using user input or camera defaults.
    """
    if user_threshold is not None:
        return user_threshold
    if threshold_map is not None:
        if camera_name is not None:
            return threshold_map.get(camera_name, threshold_map["camera-rgb"])
        else:
            return threshold_map["camera-rgb"]
    raise ValueError(
        "Cannot retrieve the model score threshold. Please provide a user-specified threshold or a mapping from camera name to threshold."
    )


# def visualize(
#     image: np.ndarray,
#     detections: List[List[float]],
#     scale_factor_detections: float,
# ) -> np.ndarray:
#     """
#     parameter image: image on which we want to make detections
#     parameter detections: list of bounding boxes in format [x1, y1, x2, y2]
#     parameter scale_factor_detections: scale detections by the given factor to allow blurring more area, 1.15 would mean 15% scaling

#     Visualize the input image with the detections and save the output image at the given path
#     """
#     image_fg = image.copy()
#     mask_shape = (image.shape[0], image.shape[1], 1)
#     mask = np.full(mask_shape, 0, dtype=np.uint8)

#     for box in detections:
#         if scale_factor_detections != 1.0:
#             box = scale_box(
#                 box, image.shape[1], image.shape[0], scale_factor_detections
#             )
#         x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
#         w = x2 - x1
#         h = y2 - y1

#         ksize = (image.shape[0] // 2, image.shape[1] // 2)
#         image_fg[y1:y2, x1:x2] = cv2.blur(image_fg[y1:y2, x1:x2], ksize)
#         cv2.ellipse(mask, (((x1 + x2) // 2, (y1 + y2) // 2), (w, h), 0), 255, -1)

#     inverse_mask = cv2.bitwise_not(mask)
#     image_bg = cv2.bitwise_and(image, image, mask=inverse_mask)
#     image_fg = cv2.bitwise_and(image_fg, image_fg, mask=mask)
#     image = cv2.add(image_bg, image_fg)


#     return image
def visualize(
    image: np.ndarray,
    detections: List[List[float]],
    scale_factor_detections: float,
    upper_face_ratio: float = 0.6,  # blur nose up
) -> np.ndarray:
    """
    Blur only the upper part of detected faces using a rectangular blur.

    Parameters:
        image: BGR image
        detections: list of bounding boxes [x1, y1, x2, y2]
        scale_factor_detections: bbox scaling factor
        upper_face_ratio: fraction of face height to blur
    """
    h_img, w_img, _ = image.shape
    image_out = image.copy()

    for box in detections:
        # Scale bbox if requested
        if scale_factor_detections != 1.0:
            box = scale_box(box, w_img, h_img, scale_factor_detections)

        x1, y1, x2, y2 = map(int, box[:4])

        # Clamp to image bounds
        x1 = max(0, min(x1, w_img - 1))
        x2 = max(0, min(x2, w_img))
        y1 = max(0, min(y1, h_img - 1))
        y2 = max(0, min(y2, h_img))

        if x2 <= x1 or y2 <= y1:
            continue

        # 🔑 Nose-up blur: keep only upper part
        face_height = y2 - y1
        blur_y2 = y1 + int(face_height * upper_face_ratio)

        blur_y2 = min(blur_y2, y2)
        if blur_y2 <= y1:
            continue

        roi = image_out[y1:blur_y2, x1:x2]
        if roi.size == 0:
            continue

        # Reasonable rectangular blur kernel (odd numbers)
        k_w = max(15, (x2 - x1) // 3 | 1)
        k_h = max(15, (blur_y2 - y1) // 3 | 1)

        blurred = cv2.blur(roi, (k_w, k_h))
        image_out[y1:blur_y2, x1:x2] = blurred

    return image_out


def visualize_image(
    input_image_path: str,
    face_detector: Optional[EgoblurDetector],
    lp_detector: Optional[EgoblurDetector],
    output_image_path: str,
    scale_factor_detections: float,
):
    """
    parameter input_image_path: absolute path to the input image
    parameter face_detector: face detector helper (may be None)
    parameter lp_detector: license plate detector helper (may be None)
    parameter output_image_path: absolute path where the visualized image will be saved
    parameter scale_factor_detections: scale detections by the given factor to allow blurring more area

    Perform detections on the input image and save the output image at the given path.
    """
    bgr_image = read_image(input_image_path)
    image = bgr_image.copy()

    image_tensor = get_image_tensor(bgr_image)
    detections = []

    # Speed tracking variables
    total_inference_time: float = 0.0

    with patch_instances(fields=PATCH_INSTANCES_FIELDS):
        # get face detections
        if face_detector is not None:
            face_results = face_detector.run(image_tensor)
            total_inference_time += face_detector.last_inference_time
            if face_results:
                if len(face_results) != 1:
                    raise ValueError(
                        f"EgoblurDetector.run is expected to return results for a single "
                        f"image in this script, got {len(face_results)}."
                    )
                detections.extend(face_results[0])

        # get license plate detections
        if lp_detector is not None:
            lp_results = lp_detector.run(image_tensor)
            total_inference_time += lp_detector.last_inference_time
            if lp_results:
                if len(lp_results) != 1:
                    raise ValueError(
                        f"EgoblurDetector.run is expected to return results for a single "
                        f"image in this script, got {len(lp_results)}."
                    )
                detections.extend(lp_results[0])

    blur_start_time = time.time()
    image = visualize(
        image,
        detections,
        scale_factor_detections,
    )
    blur_end_time = time.time()
    blur_time = blur_end_time - blur_start_time

    # Print speed report
    logger.info("=" * 60)
    logger.info("SPEED REPORT (Image)")
    logger.info("=" * 60)
    logger.info(f"Inference time: {total_inference_time:.4f} seconds")
    logger.info(f"Blurring time:  {blur_time:.4f} seconds")
    logger.info(f"Total time:     {total_inference_time + blur_time:.4f} seconds")
    logger.info("=" * 60)

    write_image(image, output_image_path)


def _create_detector(
    model_path: Optional[str],
    device: str,
    detection_class: ClassID,
    score_threshold: float,
    nms_iou_threshold: float,
) -> Optional[EgoblurDetector]:
    """Create an EgoblurDetector on a specific device, or None if model_path is None."""
    if model_path is None:
        return None
    return EgoblurDetector(
        model_path=model_path,
        device=device,
        detection_class=detection_class,
        score_threshold=score_threshold,
        nms_iou_threshold=nms_iou_threshold,
        resize_aug={
            "min_size_test": RESIZE_MIN_GEN2,
            "max_size_test": RESIZE_MAX_GEN2,
        },
    )


def _process_frame_on_gpu(
    frame_idx: int,
    bgr_image: np.ndarray,
    face_detector: Optional[EgoblurDetector],
    lp_detector: Optional[EgoblurDetector],
    scale_factor_detections: float,
    device: str,
) -> Tuple[int, np.ndarray, float, float]:
    """
    Process a single frame using detectors bound to a specific GPU.
    Returns (frame_idx, visualized_rgb, inference_time, blur_time).
    """
    # Build the image tensor on the target device
    bgr_transposed = np.transpose(bgr_image, (2, 0, 1))
    image_tensor = torch.from_numpy(bgr_transposed).to(device)

    detections: List[List[float]] = []
    inference_time: float = 0.0

    if face_detector is not None:
        face_results = face_detector.run(image_tensor)
        inference_time += face_detector.last_inference_time
        if face_results:
            detections.extend(face_results[0])

    if lp_detector is not None:
        lp_results = lp_detector.run(image_tensor)
        inference_time += lp_detector.last_inference_time
        if lp_results:
            detections.extend(lp_results[0])

    blur_start = time.time()
    visualized_bgr = visualize(
        bgr_image.copy(), detections, scale_factor_detections, 0.6
    )
    blur_time = time.time() - blur_start

    if visualized_bgr.dtype != np.uint8:
        visualized_bgr = np.clip(visualized_bgr, 0, 255).astype(np.uint8)

    visualized_rgb = cv2.cvtColor(visualized_bgr, cv2.COLOR_BGR2RGB)
    return frame_idx, np.ascontiguousarray(visualized_rgb), inference_time, blur_time


def visualize_video(
    input_video_path: str,
    face_detector: Optional[EgoblurDetector],
    lp_detector: Optional[EgoblurDetector],
    output_video_path: str,
    scale_factor_detections: float,
    face_model_path: Optional[str] = None,
    lp_model_path: Optional[str] = None,
    face_threshold: float = 0.5,
    lp_threshold: float = 0.5,
    nms_iou_threshold: float = 0.5,
    num_gpus: int = 1,
) -> None:
    """
    parameter input_video_path: absolute path to the input video
    parameter face_detector: face detector helper on the primary device (may be None)
    parameter lp_detector: license plate detector helper on the primary device (may be None)
    parameter output_video_path: absolute path where the visualized video will be saved
    parameter scale_factor_detections: scale detections by the given factor to allow blurring more area
    parameter face_model_path: path to face model (needed to create replicas for multi-GPU)
    parameter lp_model_path: path to LP model (needed to create replicas for multi-GPU)
    parameter face_threshold: face score threshold
    parameter lp_threshold: LP score threshold
    parameter nms_iou_threshold: NMS IoU threshold
    parameter num_gpus: number of GPUs to use
    FPS for the output video is preserved from the input video when available.

    Perform detections on the input video and save the output video at the given path.
    """
    # ── Determine devices to use ──────────────────────────────────────
    all_devices = get_all_devices()
    num_gpus = min(num_gpus, len(all_devices))
    devices = all_devices[:num_gpus]
    use_multi_gpu = len(devices) > 1 and devices[0] != "cpu"

    if use_multi_gpu:
        logger.info(f"Multi-GPU enabled: using {len(devices)} devices {devices}")
    else:
        logger.info(f"Using single device: {devices[0]}")

    # ── Build per-GPU detector replicas ───────────────────────────────
    # Index 0 reuses the detectors already created in main()
    gpu_face_detectors: List[Optional[EgoblurDetector]] = [face_detector]
    gpu_lp_detectors: List[Optional[EgoblurDetector]] = [lp_detector]

    if use_multi_gpu:
        with patch_instances(fields=PATCH_INSTANCES_FIELDS):
            for dev in devices[1:]:
                gpu_face_detectors.append(
                    _create_detector(
                        face_model_path,
                        dev,
                        ClassID.FACE,
                        face_threshold,
                        nms_iou_threshold,
                    )
                )
                gpu_lp_detectors.append(
                    _create_detector(
                        lp_model_path,
                        dev,
                        ClassID.LICENSE_PLATE,
                        lp_threshold,
                        nms_iou_threshold,
                    )
                )

    # ── Read video metadata ───────────────────────────────────────────
    video_reader_clip = VideoFileClip(input_video_path)
    input_fps = getattr(video_reader_clip, "fps", None)

    if not (
        isinstance(input_fps, (int, float))
        and math.isfinite(input_fps)
        and input_fps > 0
    ):
        raise ValueError(
            f"Input video FPS is unavailable or invalid for {input_video_path}. "
            "Please provide a video file with a valid fixed FPS."
        )
    output_fps = float(input_fps)

    # Speed tracking variables
    total_inference_time: float = 0.0
    total_blur_time: float = 0.0
    total_frame_time: float = 0.0
    frame_count: int = 0
    wall_start = time.time()

    # ── Estimate total frames for progress bar ────────────────────────
    total_frames: Optional[int] = None
    reader = getattr(video_reader_clip, "reader", None)
    if reader is not None:
        nframes = getattr(reader, "nframes", None)
        if isinstance(nframes, (int, float)) and math.isfinite(nframes) and nframes > 0:
            total_frames = int(nframes)
    if total_frames is None:
        duration = getattr(video_reader_clip, "duration", None)
        if isinstance(duration, (int, float)):
            estimated_total = duration * output_fps
            if math.isfinite(estimated_total) and estimated_total > 0:
                total_frames = int(estimated_total)

    # ── Process frames ────────────────────────────────────────────────
    visualized_frames: List[Optional[np.ndarray]] = []

    try:
        with patch_instances(fields=PATCH_INSTANCES_FIELDS):
            frame_iterator = video_reader_clip.iter_frames(fps=output_fps)

            if use_multi_gpu:
                # --- Multi-GPU path: thread pool, round-robin across GPUs ---
                # Pre-read all frames (needed to allow parallel submission)
                raw_frames: List[np.ndarray] = []
                read_iter = frame_iterator
                if tqdm is not None:
                    read_iter = tqdm(
                        frame_iterator,
                        total=total_frames,
                        desc="Reading frames",
                        unit="frame",
                    )
                for frame in read_iter:
                    if frame.ndim == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                    raw_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if tqdm is not None and hasattr(read_iter, "close"):
                    read_iter.close()

                frame_count = len(raw_frames)
                visualized_frames = [None] * frame_count

                n_workers = len(devices)
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futures = {}
                    for idx, bgr_image in enumerate(raw_frames):
                        gpu_idx = idx % n_workers
                        fut = pool.submit(
                            _process_frame_on_gpu,
                            idx,
                            bgr_image,
                            gpu_face_detectors[gpu_idx],
                            gpu_lp_detectors[gpu_idx],
                            scale_factor_detections,
                            devices[gpu_idx],
                        )
                        futures[fut] = idx

                    pbar = None
                    if tqdm is not None:
                        pbar = tqdm(
                            total=frame_count,
                            desc="Inferring (multi-GPU)",
                            unit="frame",
                        )
                    for fut in as_completed(futures):
                        fidx, vis_rgb, inf_t, blur_t = fut.result()
                        visualized_frames[fidx] = vis_rgb
                        total_inference_time += inf_t
                        total_blur_time += blur_t
                        if pbar is not None:
                            pbar.update(1)
                    if pbar is not None:
                        pbar.close()

            else:
                # --- Single-GPU path (original behaviour) ---
                progress_iterator = frame_iterator
                if tqdm is not None:
                    progress_iterator = tqdm(
                        frame_iterator,
                        total=total_frames,
                        desc="Processing frames",
                        unit="frame",
                    )

                try:
                    for frame in progress_iterator:
                        frame_start_time = time.time()

                        if frame.ndim == 2:
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

                        bgr_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        image_tensor = get_image_tensor(bgr_image)
                        detections: List[List[float]] = []

                        frame_inference_time: float = 0.0

                        if face_detector is not None:
                            face_results = face_detector.run(image_tensor)
                            frame_inference_time += face_detector.last_inference_time
                            if face_results:
                                if len(face_results) != 1:
                                    raise ValueError(
                                        "EgoblurDetector.run is expected to return results "
                                        f"for a single image in this script, got {len(face_results)}."
                                    )
                                detections.extend(face_results[0])
                        if lp_detector is not None:
                            lp_results = lp_detector.run(image_tensor)
                            frame_inference_time += lp_detector.last_inference_time
                            if lp_results:
                                if len(lp_results) != 1:
                                    raise ValueError(
                                        "EgoblurDetector.run is expected to return results "
                                        f"for a single image in this script, got {len(lp_results)}."
                                    )
                                detections.extend(lp_results[0])

                        total_inference_time += frame_inference_time

                        blur_start_time = time.time()
                        visualized_bgr = visualize(
                            bgr_image.copy(), detections, scale_factor_detections, 0.7
                        )
                        blur_end_time = time.time()
                        total_blur_time += blur_end_time - blur_start_time

                        if visualized_bgr.dtype != np.uint8:
                            visualized_bgr = np.clip(visualized_bgr, 0, 255).astype(
                                np.uint8
                            )

                        visualized_rgb = cv2.cvtColor(visualized_bgr, cv2.COLOR_BGR2RGB)
                        visualized_frames.append(np.ascontiguousarray(visualized_rgb))

                        frame_end_time = time.time()
                        total_frame_time += frame_end_time - frame_start_time
                        frame_count += 1
                finally:
                    if tqdm is not None and hasattr(progress_iterator, "close"):
                        progress_iterator.close()
    finally:
        video_reader_clip.close()

    if not visualized_frames or (
        use_multi_gpu and any(f is None for f in visualized_frames)
    ):
        raise ValueError(
            f"No frames were processed from {input_video_path}. "
            "Please verify the input video file."
        )

    wall_elapsed = time.time() - wall_start

    # Print speed report
    if frame_count > 0:
        avg_inference_spf = total_inference_time / frame_count
        avg_blur_spf = total_blur_time / frame_count

        logger.info("=" * 60)
        logger.info("SPEED REPORT")
        logger.info("=" * 60)
        logger.info(f"GPUs used: {len(devices)} {devices}")
        logger.info(f"Total frames processed: {frame_count}")
        logger.info(f"Avg inference time: {avg_inference_spf:.4f} seconds/frame")
        logger.info(f"Avg blurring time:  {avg_blur_spf:.4f} seconds/frame")
        logger.info("-" * 60)
        logger.info(f"Total inference time: {total_inference_time:.2f} seconds")
        logger.info(f"Total blur time:      {total_blur_time:.2f} seconds")
        logger.info(f"Wall-clock time:      {wall_elapsed:.2f} seconds")
        logger.info(f"Effective FPS:        {frame_count / wall_elapsed:.1f}")
        logger.info("=" * 60)

    clip = ImageSequenceClip(visualized_frames, fps=output_fps)
    audio_source = None
    try:
        # Preserve original audio if present
        audio_source = VideoFileClip(input_video_path)
        original_audio = audio_source.audio
        if original_audio is not None:
            clip = clip.set_audio(original_audio)
            logger.info("Original audio track found and will be preserved.")
        else:
            logger.info("No audio track found in the input video.")

        clip.write_videofile(
            output_video_path,
            codec="libx264",
            fps=output_fps,
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        )
        logger.info(f"Successfully output video to:{output_video_path}")
    finally:
        if audio_source is not None:
            audio_source.close()
        clip.close()


# def visualize_video(
#     input_video_path: str,
#     face_detector: Optional[EgoblurDetector],
#     lp_detector: Optional[EgoblurDetector],
#     output_video_path: str,
#     scale_factor_detections: float,
# ) -> None:
#     """
#     Same as before, but faces are blurred only from nose up.
#     """
#     visualized_frames: List[np.ndarray] = []
#     video_reader_clip = VideoFileClip(input_video_path)
#     input_fps = getattr(video_reader_clip, "fps", None)

#     if not (
#         isinstance(input_fps, (int, float))
#         and math.isfinite(input_fps)
#         and input_fps > 0
#     ):
#         raise ValueError(
#             f"Input video FPS is unavailable or invalid for {input_video_path}. "
#             "Please provide a video file with a valid fixed FPS."
#         )

#     output_fps = float(input_fps)

#     total_inference_time = 0.0
#     total_blur_time = 0.0
#     total_frame_time = 0.0
#     frame_count = 0

#     # 🔑 ratio of face height to blur (nose up)
#     UPPER_FACE_RATIO = 1

#     try:
#         with patch_instances(fields=PATCH_INSTANCES_FIELDS):
#             frame_iterator = video_reader_clip.iter_frames(fps=output_fps)

#             progress_iterator = (
#                 tqdm(frame_iterator, desc="Processing frames", unit="frame")
#                 if tqdm is not None
#                 else frame_iterator
#             )

#             try:
#                 for frame in progress_iterator:
#                     frame_start_time = time.time()

#                     if frame.ndim == 2:
#                         frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

#                     bgr_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#                     image_tensor = get_image_tensor(bgr_image)

#                     detections: List[List[float]] = []
#                     frame_inference_time = 0.0

#                     # ---------------- FACE DETECTIONS ----------------
#                     if face_detector is not None:
#                         face_results = face_detector.run(image_tensor)
#                         frame_inference_time += face_detector.last_inference_time

#                         if face_results:
#                             for det in face_results[0]:
#                                 x1, y1, x2, y2 = det[:4]

#                                 face_height = y2 - y1
#                                 new_y2 = y1 + UPPER_FACE_RATIO * face_height

#                                 # Create modified detection (nose up only)
#                                 new_det = det.copy()
#                                 new_det[3] = new_y2  # y2

#                                 detections.append(new_det)

#                     # ---------------- LICENSE PLATES (unchanged) ----------------
#                     if lp_detector is not None:
#                         lp_results = lp_detector.run(image_tensor)
#                         frame_inference_time += lp_detector.last_inference_time

#                         if lp_results:
#                             detections.extend(lp_results[0])

#                     total_inference_time += frame_inference_time

#                     # ---------------- BLUR ----------------
#                     blur_start_time = time.time()
#                     visualized_bgr = visualize(
#                         bgr_image.copy(),
#                         detections,
#                         scale_factor_detections,
#                         0.66,  # upper_face_ratio
#                     )
#                     total_blur_time += time.time() - blur_start_time

#                     if visualized_bgr.dtype != np.uint8:
#                         visualized_bgr = np.clip(visualized_bgr, 0, 255).astype(np.uint8)

#                     visualized_rgb = cv2.cvtColor(visualized_bgr, cv2.COLOR_BGR2RGB)
#                     visualized_frames.append(np.ascontiguousarray(visualized_rgb))

#                     total_frame_time += time.time() - frame_start_time
#                     frame_count += 1

#             finally:
#                 if tqdm is not None and hasattr(progress_iterator, "close"):
#                     progress_iterator.close()
#     finally:
#         video_reader_clip.close()

#     if not visualized_frames:
#         raise ValueError("No frames were processed.")

#     clip = ImageSequenceClip(visualized_frames, fps=output_fps)
#     try:
#         clip.write_videofile(
#             output_video_path,
#             codec="libx264",
#             audio=False,
#             fps=output_fps,
#             ffmpeg_params=["-pix_fmt", "yuv420p"],
#         )
#     finally:
#         clip.close()


def main() -> int:
    args = validate_inputs(parse_args())
    device = get_device()
    print(f"Using device: {device}")

    face_threshold = _get_threshold(
        args.camera_name, args.face_model_score_threshold, FACE_THRESHOLDS_GEN2
    )
    lp_threshold = _get_threshold(
        args.camera_name, args.lp_model_score_threshold, LP_THRESHOLDS_GEN2
    )

    face_detector: Optional[EgoblurDetector]
    if args.face_model_path is not None:
        face_detector = EgoblurDetector(
            model_path=args.face_model_path,
            device=device,
            detection_class=ClassID.FACE,
            score_threshold=face_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
            resize_aug={
                "min_size_test": RESIZE_MIN_GEN2,
                "max_size_test": RESIZE_MAX_GEN2,
            },
        )
    else:
        face_detector = None

    lp_detector: Optional[EgoblurDetector]
    if args.lp_model_path is not None:
        lp_detector = EgoblurDetector(
            model_path=args.lp_model_path,
            device=device,
            detection_class=ClassID.LICENSE_PLATE,
            score_threshold=lp_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
            resize_aug={
                "min_size_test": RESIZE_MIN_GEN2,
                "max_size_test": RESIZE_MAX_GEN2,
            },
        )
    else:
        lp_detector = None

    if args.input_image_path is not None:
        visualize_image(
            args.input_image_path,
            face_detector,
            lp_detector,
            args.output_image_path,
            args.scale_factor_detections,
        )

    if args.input_video_path is not None:
        visualize_video(
            args.input_video_path,
            face_detector,
            lp_detector,
            args.output_video_path,
            args.scale_factor_detections,
            face_model_path=args.face_model_path,
            lp_model_path=args.lp_model_path,
            face_threshold=face_threshold,
            lp_threshold=lp_threshold,
            nms_iou_threshold=args.nms_iou_threshold,
            num_gpus=args.num_gpus,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
