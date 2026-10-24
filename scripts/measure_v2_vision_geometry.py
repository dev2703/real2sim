"""Local pretrained detector/segmenter measurements on preselected raw source frames.

Masks are model predictions, not metric ground truth. Pixel measurements and source
provenance are retained separately from conditional geometric interpretations.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Sam2Model,
    Sam2Processor,
)

from real2sim.data.raw import sha256_file

DETECTOR = "IDEA-Research/grounding-dino-tiny"
DETECTOR_REV = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
SEGMENTER = "facebook/sam2.1-hiera-tiny"
SEGMENTER_REV = "de431c4043854a71d8101e17995dfe596bf101a5"
PROMPT = "a tennis ball. a bowl. a black mat. a white paper."


def mask_measurements(mask):
    mask = np.asarray(mask, np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {"visible_mask": False}
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    area, perimeter = cv2.contourArea(c), cv2.arcLength(c, True)
    m = cv2.moments(c)
    result = {"visible_mask": True, "area_px2": float(area),
              "bbox_xyxy_px": [x, y, x + w - 1, y + h - 1],
              "centroid_xy_px": [m["m10"] / m["m00"], m["m01"] / m["m00"]] if m["m00"] else None,
              "circularity": float(4 * np.pi * area / perimeter ** 2) if perimeter else 0.,
              "touches_image_boundary": bool(x == 0 or y == 0 or x + w == mask.shape[1] or y + h == mask.shape[0]),
              "outer_contour_source_px": c[::max(1, len(c) // 240), 0].astype(float).tolist()}
    if len(c) >= 5:
        center, axes, angle = cv2.fitEllipse(c)
        result["visible_silhouette_ellipse"] = {"center_xy_px": list(center), "diameters_px": list(axes), "angle_deg": angle,
            "interpretation": "Fit to visible external contour; bowl silhouette is not its rim, and partial ball silhouettes are not full sphere boundaries."}
    return result


def run(selection_path: Path, out: Path, camera: str = "both"):
    if out.exists():
        raise FileExistsError(out)
    selection = json.loads(selection_path.read_text())
    detector_path = Path(".cache/pretrained/IDEA-Research--grounding-dino-tiny-a2bb814dd30d")
    segmenter_path = Path(".cache/pretrained/facebook--sam2.1-hiera-tiny-de431c404385")
    torch.set_num_threads(2)
    processor = AutoProcessor.from_pretrained(detector_path, local_files_only=True)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(detector_path, local_files_only=True).eval()
    sam_processor = Sam2Processor.from_pretrained(segmenter_path, local_files_only=True)
    sam = Sam2Model.from_pretrained(segmenter_path, local_files_only=True).eval()
    out.mkdir(parents=True)
    report = {"schema": "source_frame_grounded_segmentation_v1", "selection_path": str(selection_path),
              "selection_sha256": sha256_file(selection_path), "device": "cpu", "torch_threads": 2,
              "detector": {"repo": DETECTOR, "revision": DETECTOR_REV, "weights_sha256": sha256_file(detector_path / "model.safetensors"),
                           "prompt": PROMPT, "box_threshold": .22, "text_threshold": .22},
              "segmenter": {"repo": SEGMENTER, "revision": SEGMENTER_REV, "weights_sha256": sha256_file(segmenter_path / "model.safetensors"),
                             "prompts": "Highest-scoring detector box per named class; additional iPhone foot box is a declared visual prompt, not an independently verified CAD identity."},
              "reserved_episodes_used": [], "frames": [], "physical_calibration_accepted": False,
              "script_sha256": sha256_file(Path(__file__)),
              "limitations": "Detector scores and segmenter predicted IoU are model scores, not calibrated metric uncertainty. Segmentation may fail on occlusion/partial objects. Mass, friction, force and hidden jaw gaps cannot be measured from these RGB captures alone."}
    for source in selection["images"]:
        if camera != "both" and source["camera_key"].rsplit(".", 1)[-1] != camera:
            continue
        path = Path(source["path"])
        if sha256_file(path) != source["sha256"]:
            raise ValueError(f"source image checksum mismatch: {path}")
        image = Image.open(path).convert("RGB")
        started = time.perf_counter()
        inputs = processor(images=image, text=PROMPT, return_tensors="pt")
        with torch.inference_mode():
            outputs = detector(**inputs)
        detections = processor.post_process_grounded_object_detection(outputs, inputs.input_ids,
            threshold=.22, text_threshold=.22, target_sizes=[image.size[::-1]])[0]
        candidates = [{"label": label, "score": float(score), "box_source_xyxy": box.tolist()}
                      for label, score, box in zip(detections["text_labels"], detections["scores"], detections["boxes"], strict=True)]
        best = {}
        for candidate in sorted(candidates, key=lambda v: -v["score"]):
            label = candidate["label"]
            for category in ("tennis ball", "bowl", "black mat", "white paper"):
                if category in label and category not in best:
                    best[category] = candidate
        if source["camera_key"].endswith("iphone"):
            best["foot_prompt"] = {"label": "visually prompted lower white foot region", "score": None,
                                   "box_source_xyxy": [160., 325., 380., 420.]}
        masks, objects = {}, {}
        overlay = np.asarray(image).copy()
        if best:
            sam_inputs = sam_processor(images=image, input_boxes=[[v["box_source_xyxy"] for v in best.values()]], return_tensors="pt")
            with torch.inference_mode():
                output = sam(**sam_inputs, multimask_output=False)
            processed = sam_processor.post_process_masks(output.pred_masks.cpu(), sam_inputs["original_sizes"].cpu())[0]
            for j, (category, detection) in enumerate(best.items()):
                mask = processed[j, 0].numpy().astype(np.uint8)
                masks[category.replace(" ", "_")] = mask
                objects[category] = {**detection, "segmentation_predicted_iou": float(output.iou_scores[0, j, 0]), **mask_measurements(mask)}
                color = [(240, 90, 30), (30, 180, 220), (30, 220, 70), (240, 80, 230), (180, 120, 10)][j % 5]
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, 2)
                x, y = map(int, detection["box_source_xyxy"][:2])
                cv2.putText(overlay, category, (max(0, x), max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
        stem = path.stem
        mask_path = out / f"{stem}_masks.npz"
        np.savez_compressed(mask_path, **masks)
        preview = out / f"{stem}_overlay.png"
        cv2.imwrite(str(preview), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        row = {**source, "objects": objects, "all_detections": candidates,
               "mask_path": str(mask_path), "mask_sha256": sha256_file(mask_path),
               "overlay_path": str(preview), "wall_s": time.perf_counter() - started}
        report["frames"].append(row)
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(stem, list(best), f"{row['wall_s']:.2f}s", flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", type=Path, default=Path("outputs/v2/random_vision_geometry_20261024/selection.json"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--camera", choices=["iphone", "top", "both"], default="both")
    args = p.parse_args()
    result = run(args.selection, args.out, args.camera)
    print(f"Saved {len(result['frames'])} model-measured frames to {args.out}")
