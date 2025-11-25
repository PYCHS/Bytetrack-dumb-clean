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
    quiet=False,
    img_size={"img_h": 375, "img_w": 1242},
    crop_info=None,
):
    """Select target IDs from cropped images using LLM.

    Args:
        crops_dir: Directory containing cropped images named as "i{track_id}_f{frame_id}.jpg"
        prompt: Text prompt for LLM target selection
        threshold: Similarity threshold (not used in LLM-based selection)
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
            # Calculate relative position
            center_x = (bbox["x1"] + bbox["x2"]) / 2
            center_y = (bbox["y1"] + bbox["y2"]) / 2
            rel_x = center_x / img_size["img_w"]  # 0.0 = leftmost, 1.0 = rightmost

            # Determine horizontal position - be more strict about "left"
            if rel_x < 0.35:
                position_desc = "on the left side of the road"
            elif rel_x > 0.65:
                position_desc = "on the right side of the road"
            else:
                position_desc = "in the center or directly ahead"

            crop_details = (
                f"Context: This is a {crop.get_class(crop.cls)} cropped from a dashcam/traffic scene.\n"
                f"Location: {position_desc} (x={center_x:.0f}/{img_size['img_w']}px = {rel_x*100:.0f}% from left)\n\n"
            )
        else:
            crop_details = (
                f"Context: This is a person cropped from a traffic scene.\n\n"
            )
        # print(f"crop_details: {crop_details}")

        payload = {
            "model": "qwen2.5vl",
            "prompt": (
                f"{crop_details}"
                f"Question: {prompt}\n\n"
                f"Consider both the {crop.get_class(crop.cls)}'s appearance and its position. "
                f"Note: A vehicle 'in the center or directly ahead' should NOT be considered as 'in the left'.\n\n"
                f"Does this vehicle match the description? Answer 'yes' or 'no'."
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
