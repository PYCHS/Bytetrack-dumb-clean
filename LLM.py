import os
import requests
import base64
import json
import re
from PIL import Image
import io

API_URL = "http://localhost:11434/api/generate"

# 【新增】定義需要進行方向性判斷的 Prompt 列表
DIRECTIONAL_PROMPTS = [
    'cars-in-the-counter-direction-of-ours', 
    'cars-in-the-same-direction-of-ours'
]

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
    # 【修改】確保使用相同的格式，避免 LLM 處理錯誤
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

    # 【新增】判斷當前的 prompt 是否需要進行方向性判斷
    p = prompt.lower()
    is_directional = any(x in p for x in DIRECTIONAL_PROMPTS)


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
        # 提取物體類別 (用於提示詞)
        object_type = "object" if crop is None else crop.get_class(crop.cls)

        # ----------------------------------------------------
        # ? 【修改】根據是否為方向性問題，構建不同的 LLM 提示詞和規則
        # ----------------------------------------------------
        
        llm_prompt = ""
        target_llm_answer = None # 【新增】預期 LLM 的回答 (TOWARDS/AWAY/yes)

        if is_directional:
            # 針對 '同向'/'逆向' 問題
            
            # 1. 設置目標答案
            if prompt.lower() == 'cars-in-the-counter-direction-of-ours':
                # 逆向/迎面而來 -> 車頭對著我們
                target_llm_answer = 'TOWARDS' 
            elif prompt.lower() == 'cars-in-the-same-direction-of-ours':
                # 同向/背向我們 -> 車尾對著我們
                target_llm_answer = 'AWAY'    

            # 2. 構建 LLM 提示詞 (只問面向)
            # 【新增】這是專門用於判斷車輛面向的 Prompt
            llm_prompt = (
                f"Context: This is a crop of a {object_type} (Tracking ID: {person_id}) from a dashcam/traffic scene.\n\n"
                "You are performing a strict pose classification.\n"
                "Your task: Determine the object's facing direction.\n\n"
                "Rules:\n"
                "- IGNORE blur, lighting, noise, image quality, and unclear appearance.\n"
                "- ONLY consider the object's **FACING DIRECTION** (車頭或車尾) based on the image.\n"
                "- You MUST answer EXACTLY one of the following: 'TOWARDS', 'AWAY', or 'UNCLEAR'.\n"
                "- 'TOWARDS' means the car front/headlights are visible (迎面而來/逆向).\n"
                "- 'AWAY' means the car back/taillights are visible (背向我們/同向).\n"
                "- 'UNCLEAR' if it's side-view, too blurry, or not a car.\n"
                f"Question: Is this {object_type} facing TOWARDS or AWAY from the camera?\n\n"
                "Answer format:\n"
                "TOWARDS, AWAY, or UNCLEAR\n"
            )
            
        else:
            # 針對一般分類問題 (保留原邏輯)
            
            # 1. 設置目標答案
            target_llm_answer = 'YES' 
            
            # 2. 構建 LLM 提示詞 (保留原邏輯，包含位置描述)
            if bbox:
                # 計算相對位置 (保留原邏輯)
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
                f"Your task: Determine whether {object_type} meets the condition described in the prompt .\n\n"
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
        # 3. LLM API 呼叫 (保留原邏輯)
        # ----------------------------------------------------
        payload = {
            "model": "qwen2.5vl",
            "prompt": llm_prompt,          # <-- 直接用上面組好的 llm_prompt
            "images": [encode_image(f)],   # 圖片一樣丟進去
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
        # 4. 【修改】目標選取判斷 (根據新的邏輯修改此處)
        # ----------------------------------------------------
        
        # 清理並大寫 LLM 回應，以利判斷
        clean_response = result_text.strip().upper()
        is_selected = False
        
        if is_directional:
            # 【新增邏輯】方向性問題：檢查 LLM 回應是否符合預期的面向 (TOWARDS/AWAY)
            # 例如: 如果 prompt 是 'counter-direction'，target_llm_answer 是 'TOWARDS'
            if target_llm_answer and clean_response.startswith(target_llm_answer):
                is_selected = True
        else:
            # 【保留原邏輯】一般分類問題：檢查 LLM 回應是否包含 'YES'
            if 'YES' in clean_response:
                is_selected = True

        if is_selected:
            selected_ids.append(person_id)
            if not quiet:
                print(f"[RESULT] Target ID: {person_id} selected (LLM Answer: {clean_response})")
        elif not quiet:
            print(f"[RESULT] Track ID {person_id} skipped (LLM Answer: {clean_response})")

    return selected_ids