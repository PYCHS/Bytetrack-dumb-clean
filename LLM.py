import os
import requests
import base64
import json
import re
from PIL import Image
import io

API_URL = "http://localhost:11434/api/generate"


def encode_image(path, min_size=224):
    """Encode image to base64, resizing if too small for vision model.

    Args:
        path: Path to image file
        min_size: Minimum dimension size (default 224 for vision models)
    """
    img = Image.open(path)
    width, height = img.size

    # Resize if either dimension is too small
    if width < min_size or height < min_size:
        # Calculate new size maintaining aspect ratio
        if width < height:
            new_width = min_size
            new_height = int(height * (min_size / width))
        else:
            new_height = min_size
            new_width = int(width * (min_size / height))

        img = img.resize((new_width, new_height), Image.LANCZOS)

    # Convert to bytes
    buffer = io.BytesIO()
    img.save(buffer, format=img.format or "JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def parse_filename(fname):
    match = re.match(r"i(\d+)_f(\d+)", fname)
    if match:
        return int(match.group(1)), int(match.group(2))
    return (-1, -1)


def sort_key(x):
    # fname: the outmost filename
    fname = os.path.basename(x)
    # match i<number>_f<number>: i12_f004
    person_id, frame_id = parse_filename(fname)
    return (frame_id, person_id)


def select_targets(
    crops_dir,
    prompt,
    device="cpu",
    quiet=False,
    img_size={"img_h": 375, "img_w": 1242},
    crop_info=None,
):
    """Select target IDs from cropped images using LLM.

    Args:
        crops_dir: Directory containing cropped images named as "i{track_id}_f{frame_id}.jpg"
        prompt: Text prompt for LLM target selection
        threshold: Similarity threshold (not used in LLM-based selection)
        device: Device to use (cpu/cuda)
        quiet: Whether to suppress output

    Returns:
        List of selected track IDs
    """
    if not os.path.exists(crops_dir):
        return []

    files = sorted(
        [
            os.path.abspath(os.path.join(crops_dir, f))
            for f in os.listdir(crops_dir)
            if os.path.isfile(os.path.join(crops_dir, f))
            and (f.endswith(".jpg") or f.endswith(".png"))
        ],
        key=sort_key,  # sort frame_id first, then person_id
    )

    if not files:
        return []

    selected_ids = []

    for idx, f in enumerate(files, 1):
        fname = os.path.basename(f)
        person_id, frame_id = parse_filename(fname)

        if not quiet:
            print(f"[INFO] ({idx}/{len(files)}) Processing {fname}")

        crop = None
        bbox = None
        if crop_info:
            for crop_item in crop_info:
                # print(crop_item)
                # print(f"file_path: {f}")
                if (
                    f.replace("/home/seanachan/ByteTrack_ultralytics/", "")
                    == crop_item.crop_path
                ):
                    crop = crop_item
                    break
        # print(f.replace("/home/seanachan/ByteTrack_ultralytics/", ""))
        # print(crop_item.crop_path)
        if crop and hasattr(crop, "bbox"):
            bbox = crop.bbox
        # print(f"bbox: {bbox}")

        if bbox:
            # Calculate relative position and center
            center_x = (bbox["x1"] + bbox["x2"]) / 2
            center_y = (bbox["y1"] + bbox["y2"]) / 2
            rel_x = center_x / img_size["img_w"]
            rel_y = center_y / img_size["img_h"]

            # Determine position in image
            if rel_x < 0.33:
                horizontal_pos = "left side"
            elif rel_x < 0.67:
                horizontal_pos = "center"
            else:
                horizontal_pos = "right side"

            crop_details = (
                f"This cropped object is located at the {horizontal_pos} of the original image.\n"
                f"Detailed position: Center at ({center_x:.1f}, {center_y:.1f}), "
                f"relative position: {rel_x*100:.1f}% from left, {rel_y*100:.1f}% from top.\n"
                f"Bounding box: (x1: {bbox['x1']:.1f}, y1: {bbox['y1']:.1f}, "
                f"x2: {bbox['x2']:.1f}, y2: {bbox['y2']:.1f}).\n"
                f"Image dimensions: {img_size['img_w']}x{img_size['img_h']} pixels.\n"
            )
        else:
            crop_details = ""
        # print(f"crop_details: {crop_details}")

        payload = {
            "model": "qwen2.5vl",
            "prompt": (
                f"You are analyzing a cropped object from a traffic scene.\n\n"
                f"Image Information:\n"
                f"Original image size: {img_size['img_w']}x{img_size['img_h']} pixels\n"
                f"{crop_details}\n"
                f"{'='*40}\n"
                f"Task: {prompt}\n"
                f"{'='*40}\n\n"
                f"Based on the cropped image and its position information, determine if this object meets the requirement.\n"
                f"IMPORTANT: Pay careful attention to:\n"
                f"1. The object's position in the scene (left/center/right)\n"
                f"2. The object's appearance, characteristics, and color\n"
                f"3. Whether all conditions in the task are satisfied\n\n"
                f"Answer ONLY 'yes' or 'no' without any explanation."
            ),
            "images": [encode_image(f)],
        }

        try:
            resp = requests.post(API_URL, json=payload, stream=True)
            resp.raise_for_status()
        except Exception as e:
            if not quiet:
                print(f"[ERROR] LLM request failed: {e}")
            continue

        result_text = ""
        for line in resp.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        result_text += data["response"]
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

        if not quiet:
            print(f"[DEBUG] LLM response: {result_text.strip()}")

        if "yes" in result_text.lower():
            selected_ids.append(person_id)
            if not quiet:
                print(f"[RESULT] Target ID: {person_id} selected")

    return selected_ids
