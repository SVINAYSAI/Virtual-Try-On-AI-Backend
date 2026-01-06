# occasion_based_styling_api.py
"""
FastAPI router for outfit suggestions based on occasion and user appearance.
Returns ONLY suggestions without generating images.
Improvements:
- Optional gender support (form field) + heuristic fallback
- More robust mediapipe usage (context-managed)
- More robust skin-tone estimation (LAB + sampling)
- Better prompt that includes gender & strict JSON schema examples
- Safer text-model calls and repeated parsing attempts
- JSON validation and guaranteed 3 suggestions fallback
"""

import os
import json
import traceback
from typing import Optional, Dict, Any, List, Tuple

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

import numpy as np
import cv2
from sklearn.cluster import KMeans
import mediapipe as mp
from PIL import Image

# Vertex AI (Gemini)
from vertexai.preview.generative_models import GenerativeModel, Part

router = APIRouter(prefix="/api/styling", tags=["Occasion-Based Styling"])

# Model will be initialized after vertexai.init() is called
text_model = None

# Initialize text model lazily
def get_text_model():
    """Lazy load the text model after Vertex AI initialization"""
    global text_model
    if text_model is None:
        # Keep this lazy in case vertexai.init() hasn't run yet on startup
        text_model = GenerativeModel("gemini-2.5-flash")
    return text_model

# ---------- Utilities ----------
def mime_type_for_filename(name: str) -> str:
    ext = os.path.splitext(name)[1].lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")

def part_from_bytes(data: bytes, filename_hint: str = "file.jpg"):
    return Part.from_data(mime_type=mime_type_for_filename(filename_hint), data=data)

def extract_text_from_response(response) -> Optional[str]:
    """Extract text content from a generative response candidate"""
    try:
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return None
        first = candidates[0]
        content = getattr(first, "content", None)
        # Try several common shapes
        if hasattr(content, "text") and isinstance(content.text, str):
            return content.text
        if isinstance(content, (list, tuple)) and len(content) > 0:
            maybe = content[0]
            if hasattr(maybe, "text"):
                return maybe.text
            if isinstance(maybe, str):
                return maybe
        if hasattr(first, "output_text"):
            return first.output_text
        # Fallback to string conversion
        return str(first)
    except Exception as e:
        print("❌ extract_text_from_response error:", e)
        traceback.print_exc()
        return None

# ---------- Mediapipe helpers ----------
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

# ---------- Feature extraction (improved heuristics) ----------
def bytes_to_bgr(img_bytes: bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Unable to decode image bytes (invalid image).")
    return bgr

def _sample_pixels(img: np.ndarray, n: int = 2000) -> np.ndarray:
    flat = img.reshape(-1, 3)
    if len(flat) > n:
        idx = np.random.choice(len(flat), n, replace=False)
        return flat[idx]
    return flat

def estimate_skin_tone(bgr_img: np.ndarray) -> Dict[str, Any]:
    """
    Use LAB color space + kmeans to pick dominant skin-like color from center crop.
    Returns readable category + representative rgb.
    """
    h, w = bgr_img.shape[:2]
    # central crop (less likely to include background)
    y1, y2 = int(h * 0.25), int(h * 0.75)
    x1, x2 = int(w * 0.25), int(w * 0.75)
    crop = bgr_img[y1:y2, x1:x2]
    if crop.size == 0:
        crop = bgr_img

    # Convert to LAB for perceptual clustering
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    samples = _sample_pixels(lab, n=3000).astype(np.float32)

    try:
        km = KMeans(n_clusters=3, random_state=42, n_init=4).fit(samples)
        centers = km.cluster_centers_.astype(int)
        # Convert centers back to BGR for brightness measure
        centers_bgr = np.array([cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_LAB2BGR)[0, 0] for c in centers])
        brightness = centers_bgr.sum(axis=1)
        chosen_bgr = centers_bgr[brightness.argmax()].tolist()
    except Exception:
        # fallback to average RGB of the crop
        avg = crop.reshape(-1, 3).mean(axis=0).astype(int).tolist()
        chosen_bgr = avg

    # convert to RGB for downstream usage
    b, g, r = int(chosen_bgr[0]), int(chosen_bgr[1]), int(chosen_bgr[2])
    # Use luminance formula in sRGB
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if lum > 185:
        tone = "Very Fair"
    elif lum > 155:
        tone = "Fair"
    elif lum > 125:
        tone = "Medium"
    elif lum > 95:
        tone = "Dusky"
    else:
        tone = "Deep"

    return {"skin_tone": tone, "skin_rgb": [r, g, b]}

def estimate_face_shape(bgr_img: np.ndarray) -> Dict[str, Any]:
    img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as fm:
        res = fm.process(img_rgb)
    if not res.multi_face_landmarks:
        return {"face_shape": "Unknown", "landmark_bbox": None}
    lm = res.multi_face_landmarks[0].landmark
    xs = np.array([p.x for p in lm])
    ys = np.array([p.y for p in lm])
    width = xs.max() - xs.min()
    height = ys.max() - ys.min()
    ratio = width / (height + 1e-9)
    # More conservative thresholds
    if ratio > 0.98:
        shape = "Round"
    elif ratio < 0.78:
        shape = "Oval"
    else:
        shape = "Square"
    return {"face_shape": shape, "landmark_bbox": {"w": float(width), "h": float(height), "ratio": float(ratio)}}

def estimate_height_body(bgr_img: np.ndarray) -> Dict[str, Any]:
    img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        res = pose.process(img_rgb)
    h, w = bgr_img.shape[:2]
    height_cat = "Unknown"
    body_type = "Unknown"
    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        ys = np.array([p.y for p in lm if hasattr(p, "visibility") and getattr(p, "visibility", 0) > 0.3])
        if len(ys):
            rel = (ys.max() - ys.min())
            if rel > 0.72:
                height_cat = "Tall"
            elif rel > 0.5:
                height_cat = "Average"
            else:
                height_cat = "Petite"
        # shoulder/hip heuristic (landmark indices per mediapipe)
        try:
            ls = lm[11]; rs = lm[12]; lh = lm[23]; rh = lm[24]
            shoulder_w = abs(ls.x - rs.x) * w
            hip_w = abs(lh.x - rh.x) * w
            if hip_w <= 1e-3:
                body_type = "Unknown"
            else:
                ratio = shoulder_w / (hip_w + 1e-9)
                if ratio > 1.07:
                    body_type = "Inverted Triangle"
                elif ratio < 0.94:
                    body_type = "Triangle / Pear"
                else:
                    body_type = "Balanced / Hourglass"
        except Exception:
            body_type = "Unknown"
    else:
        # fallback based on face fraction
        fshape = estimate_face_shape(bgr_img)
        bbox = fshape.get("landmark_bbox")
        if bbox:
            face_frac = bbox["h"]
            if face_frac > 0.40:
                height_cat = "Petite"
            elif face_frac > 0.25:
                height_cat = "Average"
            else:
                height_cat = "Tall"
    return {"height_category": height_cat, "body_type": body_type}

def extract_image_features(image_bytes: bytes) -> Dict[str, Any]:
    bgr = bytes_to_bgr(image_bytes)
    skin = estimate_skin_tone(bgr)
    face = estimate_face_shape(bgr)
    body = estimate_height_body(bgr)
    features = {**skin, **face, **body}
    return features

# ---------- Gender handling ----------
def heuristic_gender_from_landmarks(bgr_img: np.ndarray) -> Optional[str]:
    """
    Very light-weight heuristic fallback if no explicit gender supplied.
    Not perfect — recommend using a small trained gender classifier in production.
    Returns: "male", "female", or None
    """
    try:
        # use face mesh ratios (jaw/cheek indicator)
        img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as fm:
            res = fm.process(img_rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        xs = np.array([p.x for p in lm])
        ys = np.array([p.y for p in lm])
        width = xs.max() - xs.min()
        height = ys.max() - ys.min()
        # Wider face relative to height sometimes leans male; smaller ratios sometimes female
        ratio = width / (height + 1e-9)
        if ratio > 0.95:
            return "male"
        if ratio < 0.82:
            return "female"
        return None
    except Exception:
        return None

def detect_gender(image_bytes: bytes, explicit_gender: Optional[str]) -> str:
    """
    Resolve gender: priority: explicit_gender form field > env-supplied model > heuristic > 'unisex'
    """
    if explicit_gender:
        g = explicit_gender.strip().lower()
        if g in ("male", "m"):
            return "male"
        if g in ("female", "f"):
            return "female"
        if g in ("unisex", "neutral", "nonbinary", "nb", "other"):
            return "unisex"

    # Optionally, support an external gender classifier if model path provided
    model_path = os.environ.get("GENDER_CLASSIFIER_PATH")
    if model_path and os.path.exists(model_path):
        try:
            # Placeholder for integration: load model and predict
            # e.g., use ONNX / torchscript classifier. We keep this as optional.
            pass
        except Exception:
            pass

    # Heuristic fallback
    try:
        bgr = bytes_to_bgr(image_bytes)
        h = heuristic_gender_from_landmarks(bgr)
        if h:
            return h
    except Exception:
        pass

    return "unisex"

# ---------- Prompt builders (improved + explicit schema + examples) ----------
def build_suggestions_prompt(occasion: str, user_text: str, features: Dict[str, Any], gender: str) -> str:
    """
    Prompt Gemini text model to return exactly 3 outfit suggestions in strict JSON format (array of 3 objects).
    Includes gender to ensure male/female-appropriate options.
    Provides a compact JSON schema + short example to reduce hallucinated formatting.
    """
    skin = features.get("skin_tone", "Unknown")
    skin_rgb = features.get("skin_rgb", [])
    face_shape = features.get("face_shape", "Unknown")
    height = features.get("height_category", "Unknown")
    body_type = features.get("body_type", "Unknown")
    short_user = (user_text or "").strip()

    # Add explicit instruction: if gender == male produce male-appropriate traditional/western suggestions,
    # if female produce female-appropriate, if unisex produce mixed options.
    gender_text = gender if gender else "unisex"

    schema = """
Return EXACTLY a JSON ARRAY of 3 objects (no commentary) with these fields in each object:
{
  "title": string,
  "why_it_suits": string (1-2 short sentences),
  "colors": [ up to 3 color names ],
  "fabric_and_details": string (brief),
  "neckline": string (or 'N/A' for male garments),
  "accessories": string,
  "footwear": string,
  "hair_makeup_tip": string (for male: grooming/hair tip),
  "image_description": string (one concise sentence describing how the garment should be placed on the uploaded photo; include color, length, drape, neckline, tight/flowy, occlusion or layering notes)
}
The array must contain exactly 3 objects. Use culturally appropriate Indian outfit types if occasion suggests (e.g., wedding -> saree/lehengas/kurta-sherwani; formal -> suit/kurta). For male gender, prefer kurta/sherwani/jodhpur/suit options where appropriate. For unisex, include mixed options.
"""

    example = """
Example single object (for format reference only — DO NOT RETURN THIS EXAMPLE):
{
  "title": "Classic Ivory Kurta Set",
  "why_it_suits": "Balanced silhouette complements average height and balanced body type; warm skin tones pair well with ivory and gold.",
  "colors": ["Ivory", "Gold"],
  "fabric_and_details": "Silk-blend kurta with subtle zari border and tonal embroidery",
  "neckline": "Mandarin collar",
  "accessories": "Gold lapel brooch, simple watch",
  "footwear": "Leather mojari or loafers",
  "hair_makeup_tip": "Neat side-part and light beard grooming",
  "image_description": "Ivory knee-length kurta with gold embroidery, slightly fitted at chest and flowing below waist, overlay from left shoulder to right hip; keep original hands/hair occlusion; place kurta over current upper-body clothing."
}
"""

    prompt = f"""
You are an expert Indian ethnic and contemporary wardrobe stylist. Produce outfit suggestions tailored to the user.

Context:
- Occasion: {occasion}
- User note: {short_user if short_user else "N/A"}
- Gender: {gender_text}
- User appearance (estimated):
  - skin_tone: {skin} (RGB: {skin_rgb})
  - face_shape: {face_shape}
  - height_category: {height}
  - body_type: {body_type}

{schema}

{example}

Return ONLY the JSON array (no explanation, no markdown). Ensure it is valid JSON parseable by a standard JSON parser.
"""
    return prompt

# ---------- JSON validation helpers ----------
def validate_suggestions_json(obj: Any) -> Tuple[bool, Optional[str]]:
    """
    Validate that obj is a list of exactly 3 dicts and each dict contains required fields.
    Returns (is_valid, error_message)
    """
    required = {"title", "why_it_suits", "colors", "fabric_and_details", "neckline",
                "accessories", "footwear", "hair_makeup_tip", "image_description"}
    if not isinstance(obj, list):
        return False, "Top-level JSON is not a list."
    if len(obj) != 3:
        return False, f"JSON array length is {len(obj)} but expected exactly 3."
    for i, item in enumerate(obj):
        if not isinstance(item, dict):
            return False, f"Item {i} is not an object."
        missing = required - set(item.keys())
        if missing:
            return False, f"Item {i} missing fields: {sorted(list(missing))}"
        # basic types
        if not isinstance(item["colors"], list):
            return False, f"Item {i} field 'colors' is not a list."
    return True, None

# ---------- Endpoint: styling suggestions ONLY (no image generation) ----------
@router.post("/suggestions")
async def get_styling_suggestions(
    avatar: UploadFile = File(..., description="Person's photo (JPG, PNG, GIF, WebP)"),
    occasion: str = Form(..., description="Occasion (wedding, party, casual, formal, traditional, etc.)"),
    user_text: Optional[str] = Form(None, description="Additional preferences or notes from user"),
    gender: Optional[str] = Form(None, description="Optional: 'male'|'female'|'unisex' - if omitted, system will try to infer")
):
    """
    Get outfit suggestions based on occasion and user appearance.
    This endpoint ONLY provides suggestions - no image generation.
    """
    try:
        if get_text_model() is None:
            raise HTTPException(status_code=500, detail="Text model not initialized on server")

        # Read avatar
        avatar_bytes = await avatar.read()
        if not avatar_bytes:
            raise HTTPException(status_code=400, detail="Empty avatar image")

        # Detect gender (explicit > classifier > heuristic > unisex)
        resolved_gender = detect_gender(avatar_bytes, gender)
        print(f"🔍 Resolved gender: {resolved_gender}")

        # Extract features from avatar
        print(f"🔍 Analyzing user appearance from avatar...")
        features = extract_image_features(avatar_bytes)
        print(f"✅ Features extracted: {json.dumps(features, indent=2)}")

        # Build suggestions prompt
        prompt = build_suggestions_prompt(
            occasion=occasion,
            user_text=user_text or "",
            features=features,
            gender=resolved_gender
        )

        # Robust model call + parsing attempts
        text_response = None
        suggestions_text = None
        last_err = None

        for attempt in range(2):
            try:
                print(f"📤 Requesting outfit suggestions from Gemini (attempt {attempt+1})...")
                text_response = get_text_model().generate_content([prompt])
                suggestions_text = extract_text_from_response(text_response)
                if suggestions_text:
                    break
            except Exception as e:
                last_err = e
                print("⚠️ Model call failed:", e)
                traceback.print_exc()

        if not suggestions_text:
            raise HTTPException(status_code=500, detail=f"No suggestions returned by text model. Last err: {last_err}")

        # Try to parse JSON robustly
        suggestions_json = None
        parse_error = None
        try:
            suggestions_json = json.loads(suggestions_text)
        except Exception as e:
            parse_error = str(e)
            # attempt to extract the first JSON array-looking substring
            try:
                import re
                m = re.search(r'(\[.*\])', suggestions_text, flags=re.S)
                if m:
                    suggestions_json = json.loads(m.group(1))
            except Exception as e2:
                parse_error += f" | fallback parse error: {e2}"

        # If still not parseable, ask the model to convert to JSON (single explicit conversion request)
        if not suggestions_json:
            print("⚠️ Could not parse JSON directly, requesting conversion from model...")
            fallback_prompt = f"""
I previously gave outfit suggestions. Convert that content INTO a VALID JSON ARRAY of 3 objects with these fields:
title, why_it_suits, colors (array), fabric_and_details, neckline, accessories, footwear, hair_makeup_tip, image_description.
Return ONLY the JSON array and nothing else.

Content:
{suggestions_text}
"""
            try:
                conv_resp = get_text_model().generate_content([fallback_prompt])
                conv_text = extract_text_from_response(conv_resp)
                suggestions_json = json.loads(conv_text)
            except Exception as e3:
                print(f"❌ Fallback parsing failed: {e3}")
                suggestions_json = None

        # Validate the JSON and ensure fallback
        valid, error = validate_suggestions_json(suggestions_json) if suggestions_json else (False, "No json produced")
        if not valid:
            print(f"⚠️ Validation failed: {error}")
            # As last resort, build 3 simple neutral suggestions derived from features & occasion
            base_color = features.get("skin_tone", "Medium")
            fallback_suggestions = []
            for i in range(3):
                fallback_suggestions.append({
                    "title": f"Fallback Outfit {i+1}",
                    "why_it_suits": f"Auto-generated fallback for {occasion}.",
                    "colors": [base_color],
                    "fabric_and_details": "Fabric suggestions not available - pick comfortable natural fabric.",
                    "neckline": "N/A" if resolved_gender == "male" else "Round",
                    "accessories": "Minimal",
                    "footwear": "Comfortable shoes",
                    "hair_makeup_tip": "Simple grooming",
                    "image_description": "Neutral outfit overlay on upper body with natural drape; keep occlusions as-is."
                })
            suggestions_json = fallback_suggestions

        # Ensure it's exactly 3 items
        if isinstance(suggestions_json, list):
            if len(suggestions_json) > 3:
                suggestions_json = suggestions_json[:3]
            elif len(suggestions_json) < 3:
                # pad with basic variants
                while len(suggestions_json) < 3:
                    suggestions_json.append({
                        "title": "Additional Suggestion",
                        "why_it_suits": f"Alternate option for {occasion}.",
                        "colors": [features.get("skin_tone", "Medium")],
                        "fabric_and_details": "Simple fabric",
                        "neckline": "Round",
                        "accessories": "Minimal",
                        "footwear": "Comfortable shoes",
                        "hair_makeup_tip": "Keep it neat",
                        "image_description": "Neutral overlay on torso area."
                    })

        # Return suggestions (WITHOUT images)
        result = {
            "success": True,
            "features": features,
            "resolved_gender": resolved_gender,
            "occasion": occasion,
            "user_text": user_text,
            "suggestions": suggestions_json,
        }

        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
