from fastapi import APIRouter, HTTPException, Query, File, UploadFile
from fastapi.responses import FileResponse
import base64
import os
import uuid
import time
from threading import Lock
from vertexai.preview.generative_models import GenerativeModel, Part

router = APIRouter(prefix="/api/gemini", tags=["Image Transformation"])

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


# =====================================================================
#   API: IMAGE TRANSFORMATION WITH PROMPT
#   Transforms an uploaded image based on text instructions
# =====================================================================
@router.post("/transform-image")
async def transform_image(
    image: UploadFile = File(..., description="Image file to transform"),
    prompt: str = Query(..., description="Instructions for image transformation. e.g., 'Change the shirt color to red', 'Add a jacket', 'Change the background to outdoor'")
):
    """
    Transform an uploaded image based on custom prompt instructions.
    
    This API allows you to modify an image of a person with detailed instructions such as:
    - Changing clothing colors or styles
    - Adding or removing accessories
    - Changing the background
    - Adjusting pose or positioning
    - Adding or modifying makeup
    - And more...
    
    Parameters:
    - image: Image file to transform (PNG, JPG, etc.)
    - prompt: Text instructions describing the desired changes
    
    Example prompts:
    - "Change the shirt to blue"
    - "Add sunglasses and a hat"
    - "Move the person to a beach background"
    - "Make the person wear a formal suit"
    - "Change hairstyle to short hair"
    
    Returns:
        Transformed image file (PNG format)
    """
    try:
        print(f"Processing transformation request with prompt: {prompt}")
        
        # Validate file
        if not image.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Read image
        image_bytes = await image.read()
        
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Image file is empty")
        
        # Get file extension
        file_ext = image.filename.split('.')[-1].lower()
        
        # Validate image format
        valid_formats = ['jpg', 'jpeg', 'png', 'gif', 'webp']
        if file_ext not in valid_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image format. Supported formats: {', '.join(valid_formats)}"
            )
        
        mime_type = f"image/{file_ext}"
        
        print(f"Image received: {image.filename} ({len(image_bytes)} bytes)")
        
        # Create image part
        image_part = Part.from_data(mime_type=mime_type, data=image_bytes)
        
        # Enhancement prompt with safety guidelines
        enhancement_prompt = (
            f"You are an expert image editor and fashion stylist. "
            f"Transform the provided image according to these instructions: {prompt}\n\n"
            f"IMPORTANT GUIDELINES:\n"
            f"- Maintain the person's identity and original features (face must remain the same)\n"
            f"- Ensure photorealistic quality and natural appearance\n"
            f"- Preserve the original pose and body composition as much as possible\n"
            f"- Keep consistent lighting and shadows with the original\n"
            f"- Do NOT change the person's body structure or face structure\n"
            f"- Maintain professional quality suitable for fashion/e-commerce\n"
            f"- Make the transformation seamless and realistic\n"
            f"\nGenerate a high-quality edited image that matches the transformation request."
        )
        
        print("Sending request to Gemini API...")
        
        # Prepare content for Gemini
        content = [image_part, enhancement_prompt]
        img_bytes = generate_with_retry(content)
        
        if img_bytes is None:
            raise HTTPException(
                status_code=500,
                detail="Transformation failed - no image returned from Gemini."
            )
        
        print(f"Transformation successful! Generated image size: {len(img_bytes)} bytes")
        
        # Create outputs directory if it doesn't exist
        os.makedirs("outputs", exist_ok=True)
        
        # Generate unique filename
        unique_id = uuid.uuid4().hex[:8]
        output_filename = f"transformed_image_{unique_id}.png"
        output_path = os.path.join("outputs", output_filename)
        
        # Save transformed image
        with open(output_path, "wb") as f:
            f.write(img_bytes)
        
        print(f"Image saved to: {output_path}")
        
        return FileResponse(
            output_path,
            media_type="image/png",
            filename=output_filename,
            headers={"Content-Disposition": f"attachment; filename={output_filename}"}
        )
    
    except HTTPException as http_err:
        print(f"HTTP Error: {http_err.detail}")
        raise http_err
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Image transformation failed: {str(e)}"
        )
