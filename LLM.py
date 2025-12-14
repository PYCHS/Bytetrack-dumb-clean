import os
import requests
import base64
import json
import re
from PIL import Image
import io

API_URL = "http://localhost:11434/api/generate"

# Dictionary mapping keywords to physical orientation
# TOWARDS = Front view (headlights)
# AWAY = Rear view (taillights)
# instead of hard code the entire prompt
DIRECTION_KEYWORDS = {
    "cars-in-the-counter-direction-of-ours": "TOWARDS",
    "cars-in-the-same-direction-of-ours": "AWAY",
    "counter direction": "TOWARDS",
    "opposite": "TOWARDS",
    "oncoming": "TOWARDS",
    "same direction": "AWAY",
    "with us": "AWAY",
    "along our way": "AWAY",
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

    # ----------------------------------------------------
    # 1. Analyze Prompt for Directionality
    # ----------------------------------------------------
    # Instead of creating a separate logic path, we generate a "Direction Hint"
    # that will be appended to the general prompt.
    prompt_lower = prompt.lower()
    direction_hint = ""
    for key, val in DIRECTION_KEYWORDS.items():
        if key in prompt_lower:
            if val == "TOWARDS":
                direction_hint = (
                    "IMPORTANT: The user is asking for 'ONCOMING/OPPOSITE' direction. "
                    "You MUST check if the vehicle's FRONT (headlights/grille) is visible. "
                    "If you see the rear/taillights, it is NOT a match."
                )
            elif val == "AWAY":
                direction_hint = (
                    "IMPORTANT: The user is asking for 'SAME DIRECTION/AWAY'. "
                    "You MUST check if the vehicle's REAR (taillights) is visible. "
                    "If you see the front/headlights, it is NOT a match."
                )
            break


    for idx, f in enumerate(files, 1):
        fname = os.path.basename(f)
        person_id, frame_id = parse_filename(fname)

        if not quiet:
            print(f"[INFO] ({idx}/{len(files)}) Processing {fname}")

        # ----------------------------------------------------
        # 2. Extract Metadata (BBox, Class)
        # ----------------------------------------------------
        crop = None
        bbox = None
        
        # Match current file with crop_info data
        if crop_info:
            for crop_item in crop_info:
                # Assuming crop_path ends with the filename
                if os.path.basename(crop_item.crop_path) == fname:
                    crop = crop_item
                    break
        
        if crop and hasattr(crop, "bbox"):
            bbox = crop.bbox

        object_type = "object" if crop is None else crop.get_class(crop.cls)

        # ----------------------------------------------------
        # 3. Calculate Spatial Position (Always)
        # ----------------------------------------------------
        position_desc = "unknown location"
        if bbox:
            center_x = (bbox["x1"] + bbox["x2"]) / 2
            rel_x = center_x / img_size["img_w"]  # 0.0 = left, 1.0 = right

            if rel_x < 0.35:
                position_desc = "on the left side of the road"
            elif rel_x > 0.65:
                position_desc = "on the right side of the road"
            else:
                position_desc = "in the center/ahead"
            
            # Add spatial context string
            location_context = f"Location: {position_desc} (Horizontal Position: {rel_x*100:.0f}%)"
        else:
            location_context = ""
        ##########
        # ----------------------------------------------------
        # 4. Construct Unified LLM Prompt
        # ----------------------------------------------------
        # We combine Object Type + Location + Direction Hint + User Prompt
        # into a single classification task.
        
        encoded_img = encode_image(f)
        if not encoded_img:
            continue

        llm_prompt = (
            f"Context: This is a crop of a {object_type} from a dashcam view.\n"
            f"{location_context}\n\n"
            f"User Prompt: \"{prompt}\"\n\n"
            f"Task: Analyze the image and determine if it strictly matches the User Prompt.\n\n"
            f"Rules:\n"
            f"1. IGNORE low resolution or blur. Focus on visible features.\n"
            f"2. {direction_hint}\n"  # Insert specific direction rules if needed
            f"3. If the User Prompt specifies color or vehicle type, check those too.\n"
            f"4. Provide your answer in JSON format.\n\n"
            f"Output Format:\n"
            f"{{\"match\": true, \"reason\": \"short explanation\"}}\n"
            f"OR\n"
            f"{{\"match\": false, \"reason\": \"short explanation\"}}"
        )

        # ----------------------------------------------------
        # 5. Call LLM API
        # ----------------------------------------------------
        payload = {
            "model": "qwen2.5vl",
            "prompt": llm_prompt,
            "images": [encoded_img],
            "temperature": 0.1, # Low temperature for consistent classification
        }

        try:
            resp = requests.post(API_URL, json=payload, stream=False) # stream=False usually easier for JSON
            resp.raise_for_status()
            
            # Handle response (Qwen API format might vary, assuming standard structure)
            # If streaming is off, we usually get the full response body
            # Adjust depending on your specific Ollama/LocalAI version behavior
            response_data = resp.json()
            result_text = response_data.get("response", "")

        except Exception as e:
            if not quiet:
                print(f"[ERROR] LLM request failed: {e}")
            continue

        if not quiet:
            print(f"[DEBUG] LLM raw response: {result_text.strip()}")

        # ----------------------------------------------------
        # 6. Parse Result
        # ----------------------------------------------------
        is_selected = False
        
        # Robust parsing for JSON boolean
        try:
            # Try to find JSON structure even if wrapped in markdown
            json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                if data.get("match") is True:
                    is_selected = True
            else:
                # Fallback: simple keyword search if JSON parsing fails
                if "true" in result_text.lower() and "match" in result_text.lower():
                    is_selected = True
        except Exception:
            # Last resort fallback
            if "true" in result_text.lower():
                is_selected = True

        if is_selected:
            selected_ids.append(person_id)
            if not quiet:
                print(f"[RESULT] Target ID: {person_id} SELECTED")
        elif not quiet:
            print(f"[RESULT] Target ID: {person_id} Skipped")

    return selected_ids