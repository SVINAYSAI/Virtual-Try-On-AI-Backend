"""
makeup_tryon_api.py
FastAPI endpoint for virtual makeup try-on using Gemini-2.5-Flash-Image.
- Accepts face image + makeup specifications
- Applies makeup virtually using Gemini AI
- Supports face, lips, eyes, and eyebrows makeup with custom colors/styles
- Returns the transformed image directly as FileResponse
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import base64
import os
import json
import time
import uuid
import traceback
from typing import Optional
from threading import Lock

from vertexai.preview.generative_models import GenerativeModel, Part

router = APIRouter(prefix="/api/gemini", tags=["Makeup Try-On"])

# Model will be initialized after vertexai.init() is called
gen_model = None

# Rate limiting configuration
REQUEST_LOCK = Lock()
LAST_REQUEST_TIME = 0
MIN_REQUEST_INTERVAL = 5  # Minimum 5 seconds between requests
MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds to wait before retry on rate limit


def get_model():
    """Lazy load the model after Vertex AI initialization"""
    global gen_model
    if gen_model is None:
        gen_model = GenerativeModel("gemini-2.5-flash-image")
    return gen_model


def throttle_request():
    """Throttle requests to avoid rate limiting"""
    global LAST_REQUEST_TIME
    with REQUEST_LOCK:
        current_time = time.time()
        time_since_last = current_time - LAST_REQUEST_TIME
        if time_since_last < MIN_REQUEST_INTERVAL:
            wait_time = MIN_REQUEST_INTERVAL - time_since_last
            print(f"⏳ Throttling: waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
        LAST_REQUEST_TIME = time.time()


def extract_image_from_response(response) -> Optional[bytes]:
    """Extract image bytes from Gemini response"""
    try:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data.mime_type.startswith("image/"):
                data = part.inline_data.data
                if isinstance(data, bytes):
                    return data
                if isinstance(data, str):
                    return base64.b64decode(data)
        return None
    except Exception as e:
        print(f"❌ Error extracting image: {e}")
        return None


def mime_type_for_filename(filename: str) -> str:
    """Determine MIME type from filename"""
    ext = os.path.splitext(filename)[1].lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    return mime_map.get(ext, "image/jpeg")


def generate_with_retry(prompt, image_parts: list = None, max_retries=MAX_RETRIES) -> Optional[bytes]:
    """
    Generate image with retry logic for rate limiting.
    
    Args:
        prompt: String prompt for image generation
        image_parts: List of Part objects (face image, etc.)
        max_retries: Number of retries on failure
    
    Returns:
        Image bytes or None
    """
    if image_parts is None:
        image_parts = []
    
    content = image_parts + [prompt] if image_parts else [prompt]
    
    for attempt in range(max_retries):
        try:
            throttle_request()
            print(f"📤 Attempt {attempt + 1}/{max_retries}: Sending request to Gemini...")
            
            response = get_model().generate_content(content)
            img_bytes = extract_image_from_response(response)
            
            if img_bytes:
                print(f"✅ Image generated successfully (size: {len(img_bytes)} bytes)")
                return img_bytes
            else:
                raise Exception("No image data in response")
                
        except Exception as e:
            error_str = str(e)
            
            # Check for rate limit error
            if "429" in error_str or "Resource exhausted" in error_str:
                if attempt < max_retries - 1:
                    wait_time = RETRY_DELAY * (attempt + 1)
                    print(f"⚠️ Rate limited (429). Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise HTTPException(
                        status_code=429,
                        detail=f"Rate limit exceeded after {max_retries} retries. Please try again in a few minutes."
                    )
            else:
                # Other errors
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    print(f"⚠️ Error: {error_str}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise HTTPException(status_code=500, detail=f"Generation failed: {error_str}")
    
    raise HTTPException(status_code=500, detail="Image generation failed after all retries")


# =====================================================================
#   API: MAKEUP VIRTUAL TRY-ON
#   Upload face image + makeup specs → returns transformed image
# =====================================================================
@router.post("/makeup-tryon")
async def makeup_tryon(
    face_image: UploadFile = File(..., description="Face image (JPG, PNG, GIF, WebP)"),
    face_makeup: Optional[str] = Form(None, description="Face makeup color/style (e.g., 'light foundation', '#FFCCBB')"),
    lip_makeup: Optional[str] = Form(None, description="Lip makeup color/style (e.g., 'red lipstick', '#FF0000')"),
    eye_makeup: Optional[str] = Form(None, description="Eye makeup color/style (e.g., 'smokey eyeshadow', 'blue eyeliner')"),
    brow_makeup: Optional[str] = Form(None, description="Eyebrow makeup color/style (e.g., 'dark brown brow pencil', '#332211')"),
    blush_makeup: Optional[str] = Form(None, description="Blush color/style (e.g., 'rosy cheeks', '#FFAAAA')"),
    highlighter_makeup: Optional[str] = Form(None, description="Highlighter color/style (e.g., 'golden glow', '#FFFFAA')"),
    contour_makeup: Optional[str] = Form(None, description="Contour color/style (e.g., 'bronze contour', '#CC8866')"),
    mascara_makeup: Optional[str] = Form(None, description="Mascara color/style (e.g., 'black volumizing mascara', '#000000')"),
    eyeliner_makeup: Optional[str] = Form(None, description="Eyeliner color/style (e.g., 'winged black eyeliner', '#000000')"),
    glitter_makeup: Optional[str] = Form(None, description="Glitter effect (e.g., 'gold glitter', 'rainbow sparkle')"),
    kajol_makeup: Optional[str] = Form(None, description="Kajol color/style (e.g., 'black kajol', 'brown kajol')"),
    concealer_makeup: Optional[str] = Form(None, description="Concealer color/style for dark circles, should blend with skin tone (e.g., 'peach concealer', '#FFCC99')"),
    custom_instructions: Optional[str] = Form(None, description="Optional custom instructions for makeup application")
):
    """
    Perform photorealistic virtual makeup try-on.
    
    Parameters:
    - face_image: Image file containing the face (JPG, PNG, GIF, WebP)
    - face_makeup: Face makeup color/style (e.g., 'light foundation', '#FFCCBB')
    - lip_makeup: Lip makeup color/style (e.g., 'red lipstick', '#FF0000')
    - eye_makeup: Eye makeup color/style (e.g., 'smokey eyeshadow', 'blue eyeliner')
    - brow_makeup: Eyebrow makeup color/style (e.g., 'dark brown brow pencil', '#332211')
    - blush_makeup: Blush color/style (e.g., 'rosy cheeks', '#FFAAAA')
    - highlighter_makeup: Highlighter color/style (e.g., 'golden glow', '#FFFFAA')
    - contour_makeup: Contour color/style (e.g., 'bronze contour', '#CC8866')
    - mascara_makeup: Mascara color/style (e.g., 'black volumizing mascara', '#000000')
    - eyeliner_makeup: Eyeliner color/style (e.g., 'winged black eyeliner', '#000000')
    - glitter_makeup: Glitter effect (e.g., 'gold glitter', 'rainbow sparkle')
    - kajol_makeup: Kajol color/style (e.g., 'black kajol', 'brown kajol')
    - concealer_makeup: Concealer color/style for dark circles, should blend with skin tone (e.g., 'peach concealer', '#FFCC99')
    - custom_instructions: Optional additional instructions for makeup application
    
    Returns:
    - PNG image with the makeup applied to the face
    
    Example:
    ```bash
    curl -X POST "http://localhost:8000/api/gemini/makeup-tryon" \
      -F "face_image=@face.jpg" \
      -F "face_makeup=natural foundation" \
      -F "lip_makeup=coral pink lipstick" \
      -F "eye_makeup=brown eyeshadow" \
      -F "brow_makeup=dark brown brows" \
      -F "blush_makeup=peach blush" \
      -F "kajol_makeup=black kajol" \
      -F "concealer_makeup=peach concealer"
    ```
    """
    try:
        # Validate file type
        valid_formats = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
        face_ext = os.path.splitext(face_image.filename)[1].lower()
        
        if face_ext not in valid_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid face image format. Supported: {', '.join(valid_formats)}"
            )
        
        # Read image file
        print(f"📥 Reading face image: {face_image.filename}")
        face_bytes = await face_image.read()
        
        if not face_bytes:
            raise HTTPException(status_code=400, detail="Empty face image file")
        
        # Create image part
        face_part = Part.from_data(
            mime_type=mime_type_for_filename(face_image.filename),
            data=face_bytes
        )
        
        # Build the makeup specification string
        makeup_specs = []
        if face_makeup:
            makeup_specs.append(f"Face: {face_makeup}")
        if lip_makeup:
            makeup_specs.append(f"Lips: {lip_makeup}")
        if eye_makeup:
            makeup_specs.append(f"Eyes: {eye_makeup}")
        if brow_makeup:
            makeup_specs.append(f"Eyebrows: {brow_makeup}")
        if blush_makeup:
            makeup_specs.append(f"Blush: {blush_makeup}")
        if highlighter_makeup:
            makeup_specs.append(f"Highlighter: {highlighter_makeup}")
        if contour_makeup:
            makeup_specs.append(f"Contour: {contour_makeup}")
        if mascara_makeup:
            makeup_specs.append(f"Mascara: {mascara_makeup}")
        if eyeliner_makeup:
            makeup_specs.append(f"Eyeliner: {eyeliner_makeup}")
        if glitter_makeup:
            makeup_specs.append(f"Glitter: {glitter_makeup}")
        if kajol_makeup:
            makeup_specs.append(f"Kajol: {kajol_makeup}")
        if concealer_makeup:
            makeup_specs.append(f"Concealer: {concealer_makeup}")
        
        if not makeup_specs:
            raise HTTPException(
                status_code=400,
                detail="At least one makeup specification must be provided"
            )
        
        makeup_description = ", ".join(makeup_specs)
        
        # Build the strict try-on prompt
        custom_inst_text = f"\nAdditional instructions: {custom_instructions}" if custom_instructions else ""
        
        prompt = f"""
You are an expert photorealistic makeup virtual try-on engine. INPUTS:
  - Image #1: face photo (DO NOT alter face structure, expression, hair, or background)
  - Makeup specifications: {makeup_description}

ABSOLUTE RULES (MUST FOLLOW):
1) Do NOT change the person's face structure, expression, hair, pose, or background.
2) Do NOT crop, reframe, resize, or alter lighting/contrast of the original photo.
3) ONLY apply the specified makeup types naturally:
   - Face makeup: Apply foundation as specified
   - Lip makeup: Apply lipstick/lip gloss as specified
   - Eye makeup: Apply eyeshadow as specified
   - Brow makeup: Apply eyebrow pencil/powder as specified
   - Blush makeup: Apply blush to cheeks as specified
   - Highlighter makeup: Apply highlighter to cheekbones/nose/eyebrows as specified
   - Contour makeup: Apply contour to jawline/cheekbones/nose as specified
   - Mascara makeup: Apply mascara to lashes as specified
   - Eyeliner makeup: Apply eyeliner to lash line as specified
   - Glitter makeup: Apply glitter effects as specified
   - Kajol makeup: Apply kajol around the eyes as specified
   - Concealer makeup: Apply specified concealer to neutralize dark circles. It MUST be translucent, feathered at the edges, and blended seamlessly into the skin's natural texture. It should look like real skin, not a solid patch.
4) Match existing lighting, shadows, and perspective so the makeup looks physically present.
5) Do NOT add text, logos, filters, or any other objects.
6) DO NOT create opaque masks, solid patches of color, or sharp edges for any makeup, especially concealer. 
7) BLENDING IS CRITICAL: All makeup edges must be feathered and soft-focus. The skin's natural pores and texture must remain visible through the makeup (translucent application).
8) Produce ONE photorealistic final image that is the original photo + the makeup applied naturally.
9) Preserve all facial features, hair, and background details exactly as they appear in the original photo.
10) The makeup should appear to be WORN ON the face naturally, not floating or misplaced.{custom_inst_text}

Return ONLY the final photorealistic image (no text, no captions).
        """.strip()
        
        print(f"🔄 Generating makeup try-on with specs: {makeup_description}")
        
        # Call Gemini with the face image
        img_bytes = generate_with_retry(
            prompt=prompt,
            image_parts=[face_part]
        )
        
        if img_bytes is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate makeup try-on image. Please try again."
            )
        
        # Save temporary output
        output_filename = f"makeup_tryon_{uuid.uuid4().hex[:8]}.png"
        output_path = os.path.join("outputs", output_filename)
        
        os.makedirs("outputs", exist_ok=True)
        
        with open(output_path, "wb") as f:
            f.write(img_bytes)
        
        print(f"✅ Makeup try-on image saved: {output_path}")
        
        # Return as file response with download header
        return FileResponse(
            output_path,
            media_type="image/png",
            filename=output_filename,
            headers={"Content-Disposition": f"attachment; filename={output_filename}"}
        )
    
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        print(f"❌ Error in makeup try-on: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))