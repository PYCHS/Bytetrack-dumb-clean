import os
import requests
import base64
import json
import re
import argparse
import sys
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
                print(crop_item)
                if crop_item.tracker_id == person_id and crop_item.frame_id == frame_id:
                    crop = crop_item
                    break

        if crop and hasattr(crop, "bbox"):
            bbox = crop.bbox

        if bbox:
            # Assume it's an object with attributes
            crop_details = (
                f"the position of this crop in the original image is (x1: {bbox.x1}, y1: {bbox.y1}, "
                f"x2: {bbox.x2}, y2: {bbox.y2}). "
            )
        else:
            crop_details = ""
        print(crop_details)

        payload = {
            "model": "qwen3-vl",
            "prompt": (
                f"The size of the original image is height: {img_size['img_h']}, "
                f"width: {img_size['img_w']}, "
                f"I will attach a cropped image from this image to you"
                f"{crop_details}\n\n"
                f"{prompt}\n"
                "Confirm whether it meets the requirement. "
                "If yes, answer 'yes'. If not, answer 'no'."
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    args = parser.parse_args()

    def get_sorted_files(crops_dir):
        return sorted(
            [
                os.path.abspath(os.path.join(crops_dir, f))
                for f in os.listdir(crops_dir)
                if os.path.isfile(os.path.join(crops_dir, f))
                and (f.endswith(".jpg") or f.endswith(".png"))
            ],
            key=sort_key,
        )

    files = get_sorted_files(args.crops)
    target_id = -1

    for idx, f in enumerate(files, 1):
        fname = os.path.basename(f)
        person_id, frame_id = parse_filename(fname)
        print(f"[INFO] ({idx}/{len(files)}) Processing {fname}")

        payload = {
            "model": "qwen2.5vl",
            "prompt": (
                f"{args.prompt}\n"
                "Confirm whether it meets the requirements. "
                "If yes, answer 'yes'. If not, answer 'no'."
            ),
            "images": [encode_image(f)],
        }

        try:
            resp = requests.post(API_URL, json=payload, stream=True)
            resp.raise_for_status()
        except Exception as e:
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

        print(f"[DEBUG] LLM response: {result_text.strip()}")

        if "yes" in result_text.lower():
            target_id = person_id
            print(f"[RESULT] Target ID: {target_id}")
            break

    print(json.dumps({"target_id": target_id}, ensure_ascii=False))
    sys.exit(0)
