import os
import requests
import base64
import json
import re
from PIL import Image
import io

API_URL = "http://localhost:11434/api/generate"

# [NEW] Define prompts that require directional judgment
DIRECTIONAL_PROMPTS = [
    'cars-in-the-counter-direction-of-ours', 
    'cars-in-the-same-direction-of-ours'
]
# Mapping from prompt keywords/full prompts to the expected LLM answer.
# Keys should be lowercase. Use both full prompt strings and shorter keyword
# variants to allow flexible matching (substring or exact).
DIRECTION_MAP = {
    'cars-in-the-counter-direction-of-ours': 'TOWARDS',
    'cars-in-the-same-direction-of-ours': 'AWAY',
    # shorter keywords to match variants in user-provided prompts
    'counter-direction': 'TOWARDS',
    'same-direction': 'AWAY',
    'counter': 'TOWARDS',
    'same': 'AWAY',
}

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
    # [MODIFIED] Force consistent image format to avoid LLM processing errors
    img.save(buffer, format=img.format if img.format in ("PNG", "JPEG") else "JPEG")
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

    # [NEW] Check whether the current prompt requires directional judgment
    # and determine the expected LLM answer using `DIRECTION_MAP`.
    p = prompt.lower()
    is_directional = any(x in p for x in DIRECTIONAL_PROMPTS)

    # Determine the expected target answer for directional prompts by
    # checking for known keywords or exact prompt matches. This avoids the
    # previous bug where substring detection and exact matching disagreed.
    target_llm_answer = None
    if is_directional:
        # First try to find a direct mapping using any key present in the
        # prompt (flexible substring match).
        for key, val in DIRECTION_MAP.items():
            if key in p:
                target_llm_answer = val
                break
        # If still not found, try exact matches with the canonical prompts.
        if target_llm_answer is None:
            for canonical in DIRECTIONAL_PROMPTS:
                if p.strip() == canonical:
                    target_llm_answer = DIRECTION_MAP.get(canonical)
                    break


    for idx, f in enumerate(files, 1):
        fname = os.path.basename(f)
        person_id, frame_id = parse_filename(fname)

        if not quiet:
            print(f"[INFO] ({idx}/{len(files)}) Processing {fname}")

        crop = None
        bbox = None
        if crop_info:
            print(crop_info)
            for crop_item in crop_info:
                file_name = os.path.basename(f)
                crop_name = os.path.basename(crop_item.crop_path)

                if file_name == crop_name:
                    crop = crop_item
                    break
        if crop and hasattr(crop, "bbox"):
            bbox = crop.bbox
        # print(f"bbox: {bbox}")
        # Extract object category (for prompt generation)
        object_type = "object" if crop is None else crop.get_class(crop.cls)

        # ----------------------------------------------------
        # [MODIFIED] Build different LLM prompts and rules
        # depending on whether direction is involved
        # ----------------------------------------------------
        
        llm_prompt = ""
        target_llm_answer = None # [NEW] Expected LLM answer (TOWARDS / AWAY / YES)

        if is_directional:
            # 2. Construct LLM prompt (only asking orientation)
            # Specialized prompt for car orientation. The LLM is instructed
            # to only return one of three tokens: TOWARDS / AWAY / UNCLEAR.
            llm_prompt = (
                f"Context: This is a crop of a {object_type} (Tracking ID: {person_id}) from a dashcam/traffic scene.\n\n"
                "You are performing a strict pose classification.\n"
                "Your task: Determine the object's facing direction.\n\n"
                "Rules:\n"
                "- IGNORE blur, lighting, noise, image quality, and unclear appearance.\n"
                "- ONLY consider the object's FACING DIRECTION (front/head or back/rear) based on the image.\n"
                "- You MUST answer EXACTLY one of the following: 'TOWARDS', 'AWAY', or 'UNCLEAR'.\n"
                "- 'TOWARDS' means the car front/headlights are visible (approaching/opposite direction).\n"
                "- 'AWAY' means the car back/taillights are visible (moving away/same direction).\n"
                "- 'UNCLEAR' if it's side-view, too blurry, or not a car.\n"
                f"Question: Is this {object_type} facing TOWARDS or AWAY from the camera?\n\n"
                "Answer format:\n"
                "TOWARDS, AWAY, or UNCLEAR\n"
            )
            
        else:
            # For general classification problems (keep original logic)

            target_llm_answer = 'YES'

            if bbox:
                # Compute relative position
                center_x = (bbox["x1"] + bbox["x2"]) / 2
                center_y = (bbox["y1"] + bbox["y2"]) / 2
                rel_x = center_x / img_size["img_w"]  # 0.0 = leftmost, 1.0 = rightmost

                # Determine horizontal position
                if rel_x < 0.35:
                    position_desc = "on the left side of the road"
                elif rel_x > 0.65:
                    position_desc = "on the right side of the road"
                else:
                    position_desc = "in the center or directly ahead"

                crop_details = (
                    f"Context: This is a {object_type} cropped from a dashcam/traffic scene.\n"
                    f"Location: {position_desc} (x={center_x:.0f}/{img_size['img_w']}px = {rel_x*100:.0f}% from left)\n\n"
                )
            else:
                crop_details = (
                    f"Context: This is a {object_type} cropped from a traffic scene.\n\n"
                )

            llm_prompt = (
                f"{crop_details}"
                "You are performing a strict binary classification.\n"
                f"Your task: Determine whether {object_type} meets the condition described in the prompt.\n\n"
                "Rules:\n"
                "- IGNORE blur, lighting, noise, image quality, and unclear appearance.\n"
                "- ONLY consider the object's HORIZONTAL POSITION (x-axis) based on the provided location description.\n"
                "- Do NOT judge based on identity or appearance.\n"
                "- Do NOT say the image is blurry.\n"
                "- You MUST answer EXACTLY one of the following: 'yes' or 'no'.\n"
                f"Question: {prompt}\n\n"
                "Answer format:\n"
                "yes or no\n"
            )

        # ----------------------------------------------------
        # 3. LLM API Call (original logic preserved)
        # ----------------------------------------------------
        payload = {
            "model": "qwen2.5vl",
            "prompt": llm_prompt,
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

        # ----------------------------------------------------
        # 4. [MODIFIED] Selection logic based on new rules
        # ----------------------------------------------------

        # Clean and uppercase the LLM response for reliable comparison
        clean_response = result_text.strip().upper()
        is_selected = False

        if is_directional:
            # [NEW LOGIC] For directional prompts: use word-boundary regex to
            # reliably detect tokens like 'TOWARDS', 'AWAY', or 'UNCLEAR'.
            # This is more robust than startswith and tolerates short
            # explanatory text from the LLM.
            if target_llm_answer:
                pattern = r"\b" + re.escape(target_llm_answer) + r"\b"
                if re.search(pattern, clean_response):
                    is_selected = True
            else:
                # Fallback: if no explicit mapping was found earlier,
                # attempt to infer selection from common keywords in the
                # prompt and the presence of TOWARDS/AWAY in the response.
                if 'counter' in p and re.search(r"\bTOWARDS\b", clean_response):
                    is_selected = True
                elif ('same' in p or 'same-direction' in p) and re.search(r"\bAWAY\b", clean_response):
                    is_selected = True
                # If UNCLEAR or no relevant token found, do not select.
        else:
            # [ORIGINAL LOGIC] For general classification: use word-boundary
            # matching for 'YES' to avoid accidental substring matches.
            if re.search(r"\bYES\b", clean_response):
                is_selected = True

        if is_selected:
            selected_ids.append(person_id)
            if not quiet:
                print(f"[RESULT] Target ID: {person_id} selected (LLM Answer: {clean_response})")
        elif not quiet:
            print(f"[RESULT] Track ID {person_id} skipped (LLM Answer: {clean_response})")

    return selected_ids