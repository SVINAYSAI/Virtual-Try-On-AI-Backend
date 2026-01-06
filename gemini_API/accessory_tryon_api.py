"""
accessory_tryon_api.py
FastAPI endpoint for minimal jewelry/accessory virtual try-on using Gemini-2.5-Flash-Image.
- Accepts image upload + accessory image upload + accessory type
- Strict prompt to ensure model only overlays the accessory without changing the person
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

router = APIRouter(prefix="/api/gemini", tags=["Accessory Try-On"])

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
        image_parts: List of Part objects (avatar, accessory images, etc.)
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
#   API: ACCESSORY VIRTUAL TRY-ON
#   Upload avatar image + accessory image → returns transformed image
# =====================================================================
@router.post("/accessory-tryon")
async def accessory_tryon(
    avatar: UploadFile = File(..., description="Person/avatar image (JPG, PNG, GIF, WebP)"),
    accessory: UploadFile = File(..., description="Accessory image (JPG, PNG, GIF, WebP)"),
    accessory_type: str = Form("necklace", description="Type of accessory: necklace, earrings, bangle, ring, bracelet, anklet, etc."),
    custom_instructions: Optional[str] = Form(None, description="Optional custom instructions (e.g., 'place on left wrist only')")
):
    """
    Perform photorealistic accessory virtual try-on.
    
    Parameters:
    - avatar: Image file containing the person (JPG, PNG, GIF, WebP)
    - accessory: Image file containing the accessory/jewelry (JPG, PNG, GIF, WebP)
    - accessory_type: Type of accessory (necklace, earrings, bangle, ring, bracelet, anklet, etc.)
    - custom_instructions: Optional additional instructions for placement
    
    Returns:
    - PNG image with the accessory overlaid onto the person
    
    Example:
    ```bash
    curl -X POST "http://localhost:8000/api/gemini/accessory-tryon" \\
      -F "avatar=@person.jpg" \\
      -F "accessory=@necklace.jpg" \\
      -F "accessory_type=necklace"
    ```
    """
    try:
        # Validate file types
        valid_formats = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
        avatar_ext = os.path.splitext(avatar.filename)[1].lower()
        accessory_ext = os.path.splitext(accessory.filename)[1].lower()
        
        if avatar_ext not in valid_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid avatar format. Supported: {', '.join(valid_formats)}"
            )
        if accessory_ext not in valid_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid accessory format. Supported: {', '.join(valid_formats)}"
            )
        
        # Read image files
        print(f"📥 Reading avatar image: {avatar.filename}")
        avatar_bytes = await avatar.read()
        
        print(f"📥 Reading accessory image: {accessory.filename}")
        accessory_bytes = await accessory.read()
        
        if not avatar_bytes or not accessory_bytes:
            raise HTTPException(status_code=400, detail="Empty image file(s)")
        
        # Create image parts
        avatar_part = Part.from_data(
            mime_type=mime_type_for_filename(avatar.filename),
            data=avatar_bytes
        )
        accessory_part = Part.from_data(
            mime_type=mime_type_for_filename(accessory.filename),
            data=accessory_bytes
        )
        
        # Build the strict try-on prompt
        custom_inst_text = f"\nAdditional instructions: {custom_instructions}" if custom_instructions else ""
        
        prompt = f"""
You are an expert photorealistic jewelry virtual try-on engine. INPUTS:
  - Image #1: person/avatar (DO NOT alter)
  - Image #2: {accessory_type} accessory to overlay

ABSOLUTE RULES (MUST FOLLOW):
1) Do NOT change the person's face, expression, hair, pose, body shape, clothing, or background.
2) Do NOT crop, reframe, resize, or alter lighting/contrast of the original photo.
3) ONLY add the {accessory_type} as a natural overlay in the correct anatomical position:
   - Necklace: hang from neck naturally
   - Earrings: place on both ears symmetrically
   - Bangle/Bracelet: place on wrist(s)
   - Ring: place on appropriate finger
   - Anklet: place on ankle
4) Match existing lighting, shadows, reflections, and perspective so the accessory looks physically present.
5) Do NOT add text, logos, filters, or any other objects.
6) Produce ONE photorealistic final image that is the original photo + the accessory blended naturally.
7) Preserve all skin, clothing, and background details exactly as they appear in the original photo.
8) The accessory should appear to be WORN ON the person naturally, not floating or misplaced.{custom_inst_text}

Return ONLY the final photorealistic image (no text, no captions).
        """.strip()
        
        print(f"🔄 Generating {accessory_type} try-on...")
        
        # Call Gemini with both images
        img_bytes = generate_with_retry(
            prompt=prompt,
            image_parts=[avatar_part, accessory_part]
        )
        
        if img_bytes is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate try-on image. Please try again."
            )
        
        # Save temporary output
        output_filename = f"tryon_{accessory_type}_{uuid.uuid4().hex[:8]}.png"
        output_path = os.path.join("outputs", output_filename)
        
        os.makedirs("outputs", exist_ok=True)
        
        with open(output_path, "wb") as f:
            f.write(img_bytes)
        
        print(f"✅ Try-on image saved: {output_path}")
        
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
        print(f"❌ Error in try-on: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
