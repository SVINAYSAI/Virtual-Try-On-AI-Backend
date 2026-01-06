from fastapi import APIRouter, HTTPException, Query, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from vertexai.preview.generative_models import GenerativeModel, Part
import base64
import os
from io import BytesIO
import json
import uuid
import time
import asyncio
from threading import Lock

router = APIRouter(prefix="/api/gemini", tags=["Gemini Model Generator"])

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
            print(f"Throttling: waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
        LAST_REQUEST_TIME = time.time()


def generate_with_retry(prompt, max_retries=MAX_RETRIES) -> bytes:
    """
    Generate image with retry logic for rate limiting.
    
    Args:
        prompt: Can be a string or list (for image + text content)
        max_retries: Number of retries on failure
    
    Returns:
        Image bytes
    """
    for attempt in range(max_retries):
        try:
            throttle_request()
            print(f"Attempt {attempt + 1}/{max_retries}: Generating image...")
            
            # Handle both string and list inputs
            if isinstance(prompt, str):
                response = get_model().generate_content([prompt])
            else:
                # Assume it's a list of content parts
                response = get_model().generate_content(prompt)
            
            img_bytes = extract_image_from_response(response)
            
            if img_bytes:
                return img_bytes
            else:
                raise Exception("No image data in response")
                
        except Exception as e:
            error_str = str(e)
            
            # Check for rate limit error
            if "429" in error_str or "Resource exhausted" in error_str:
                if attempt < max_retries - 1:
                    wait_time = RETRY_DELAY * (attempt + 1)
                    print(f"Rate limited. Waiting {wait_time} seconds before retry...")
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
                    print(f"Error: {error_str}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise HTTPException(status_code=500, detail=f"Generation failed: {error_str}")
    
    raise HTTPException(status_code=500, detail="Image generation failed after all retries")


def extract_image_from_response(response) -> bytes:
    """Extract image bytes from Gemini response"""
    try:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data.mime_type.startswith("image/"):
                data = part.inline_data.data
                return base64.b64decode(data) if isinstance(data, str) else data
        return None
    except Exception as e:
        print(f"Error extracting image: {e}")
        return None


# =====================================================================
#   API 1: ENHANCED MODEL GENERATION WITH CUSTOM PROMPT
# =====================================================================
@router.get("/generate-model")
async def generate_model(
    gender: str = Query("female", description="male / female"),
    region: str = Query("european", description="indian / european / korean / african / latino / arabic / american"),
    age_group: str = Query("young adult", description="child / teen / young adult / adult"),
    pose: str = Query("standing", description="standing / walking / side-view / portrait / mid-shot / full-shot"),
    background: str = Query("studio", description="studio / white / outdoor / beach / city / indoor / minimal"),
    outfit_type: str = Query("modern wear", description="casual / modern wear / streetwear / formal / winter / summer"),
    custom_prompt: str = Query("", description="Additional custom instructions for the model generation (optional)")
):
    """
    Generate a high-quality human fashion model with optional custom prompt.
    
    Parameters:
    - gender: male / female
    - region: indian / european / korean / african / latino / arabic / american
    - age_group: child / teen / young adult / adult
    - pose: standing / walking / side-view / portrait / mid-shot / full-shot
    - background: studio / white / outdoor / beach / city / indoor / minimal
    - outfit_type: casual / modern wear / streetwear / formal / winter / summer
    - custom_prompt: Additional text to include in the generation prompt (optional)
    """
    try:
        # Base prompt
        base_prompt = (
            "Generate a high-quality human fashion model. "
            "The model must be realistic and photorealistic with accurate skin texture, lighting, shadows, and proportions. "
            "Do NOT use sports garments. Only modern wear, daily wear, and global fashion allowed. "
            "All regions can wear all clothing types (e.g., Indian model can wear jeans and T-shirt). "
            "Generate separate shots, maintain clean realistic texture, natural shape, and real lighting."
        )

        # Dynamic parameters
        dynamic_prompt = (
            f"Model gender: {gender}. "
            f"Region/ethnicity: {region}. "
            f"Age group: {age_group}. "
            f"Pose: {pose}. "
            f"Background: {background}. "
            f"Outfit style: {outfit_type}. "
            "Full body visible. Face should be sharp, symmetrical, and detailed. "
            "Render in 19:6 reel aspect ratio. "
            "Use a natural fashion model posture."
        )

        # Combine with custom prompt if provided
        final_prompt = base_prompt + " " + dynamic_prompt
        if custom_prompt:
            final_prompt += f" {custom_prompt}"

        img_bytes = generate_with_retry(final_prompt)

        if img_bytes is None:
            raise HTTPException(status_code=500, detail="Model returned no image.")

        # Save image
        os.makedirs("outputs", exist_ok=True)
        output_path = "outputs/generated_model.png"
        with open(output_path, "wb") as f:
            f.write(img_bytes)

        return FileResponse(output_path, media_type="image/png", filename="generated_model.png")

    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
