"""
body_size_recommendation_api.py
FastAPI endpoint for body size recommendation using Gemini-2.5-Flash-Image.
- Accepts front and side view images of a person
- Takes height and gender as inputs
- Analyzes body measurements using photogrammetry principles
- Provides clothing size recommendations (T-shirt, pants, etc.)
- Returns JSON with estimated measurements and recommended sizes
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import base64
import json
import time
import traceback
from typing import Optional, Dict, Any
from threading import Lock

from vertexai.preview.generative_models import GenerativeModel, Part

router = APIRouter(prefix="/api/gemini", tags=["Body Size Recommendation"])

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


def mime_type_for_filename(filename: str) -> str:
    """Determine MIME type from filename"""
    ext = filename.lower().split('.')[-1] if '.' in filename else 'jpeg'
    mime_map = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp"
    }
    return mime_map.get(ext, "image/jpeg")


def build_body_analysis_prompt(height: float, gender: str, unit: str = "cm") -> str:
    """
    Build a flexible prompt for body size analysis.
    
    Args:
        height: Height of the person
        gender: Gender (Male/Female/Other)
        unit: Height unit (cm/inches)
    
    Returns:
        Formatted prompt string
    """
    prompt = f"""Role: You are an expert Anthropometric AI Analyst and Digital Tailor. You specialize in extracting precise body measurements from 2D images by using photogrammetry principles and reference scaling.

Task: Analyze the two attached images (Image 1: Front View and Image 2: Side View) of a person. Using the provided Height as the ground-truth reference, calculate specific body measurements and recommend clothing sizes.

Input Data:
- Image 1: Frontal full-body view
- Image 2: Side profile full-body view  
- Subject Height: {height} {unit}
- Subject Gender: {gender}

Analysis Steps:
1. Skeleton Mapping: Identify key anatomical landmarks including:
   - Acromion (shoulder tips)
   - Sternum
   - Waist (narrowest point or navel line)
   - Hips (widest point)
   - Ankles

2. Scale Calculation: In the Front View, measure the total pixel height of the subject from top of head to floor. Calculate the Pixel-to-Centimeter Ratio (Total Pixels / Known Height).

3. Horizontal Estimation: Apply this ratio to horizontal distances:
   - Shoulder width
   - Hip width
   - Waist width

4. Depth Estimation: Switch to the Side View. Re-calculate the ratio for the side view (as distance from camera may vary). Measure the depth of:
   - Chest
   - Waist
   - Hips

5. Circumference Calculation: Approximate circumferences using the Ramanujan approximation for an ellipse, where 'a' is half the width (from front view) and 'b' is half the depth (from side view).
   Formula: Circumference ≈ π [3(a+b) - √((3a+b)(a+3b))]

6. Body Shape Analysis: Determine the overall body shape category (e.g., Inverted Triangle, Rectangle, Pear, Hourglass, Apple).

7. Clothing Size Recommendations: Based on measurements, recommend international clothing sizes.

SIZING CHARTS - CRITICAL REFERENCE:

MEN'S SHIRT SIZES (Chest Circumference Primary):
- XS: Chest < 86 cm (< 34"), Shoulders < 40 cm, Sleeve 73-76 cm
- S: Chest 86-92 cm (34-36"), Shoulders 40-42 cm, Sleeve 76-79 cm
- M: Chest 92-100 cm (36-39"), Shoulders 42-44 cm, Sleeve 79-82 cm
- L: Chest 100-108 cm (39-42"), Shoulders 44-47 cm, Sleeve 82-85 cm
- XL: Chest 108-116 cm (42-45"), Shoulders 47-50 cm, Sleeve 85-88 cm
- XXL: Chest 116-124 cm (45-48"), Shoulders 50-53 cm, Sleeve 88-91 cm
- XXXL: Chest > 124 cm (> 48"), Shoulders > 53 cm, Sleeve > 91 cm

WOMEN'S SHIRT SIZES (Chest Circumference Primary):
- XS: Chest < 80 cm (< 31"), Waist < 66 cm, Shoulders < 38 cm
- S: Chest 80-87 cm (31-34"), Waist 66-73 cm, Shoulders 38-40 cm
- M: Chest 87-94 cm (34-37"), Waist 73-80 cm, Shoulders 40-42 cm
- L: Chest 94-101 cm (37-39"), Waist 80-87 cm, Shoulders 42-44 cm
- XL: Chest 101-108 cm (39-42"), Waist 87-94 cm, Shoulders 44-46 cm
- XXL: Chest 108-115 cm (42-45"), Waist 94-101 cm, Shoulders 46-48 cm

SHIRT SIZE CALCULATION ALGORITHM:
1. Calculate chest-to-waist ratio to determine fit type
2. Primary size selector: Use chest circumference as primary measurement
3. Cross-validate with: Shoulder width, waist circumference, and sleeve length
4. If measurements span multiple sizes, use the LARGER size (better to be slightly loose than too tight)
5. Fit determination:
   - Ratio (Chest-Waist) > 12cm: Regular/Relaxed fit
   - Ratio (Chest-Waist) 8-12cm: Standard fit
   - Ratio (Chest-Waist) < 8cm: Slim/Fitted
6. Size adjustment rules:
   - If shoulders are disproportionately large: Consider size UP
   - If person has narrow shoulders with large chest: Use chest-based size
   - If all measurements consistently point to different sizes: Use average and round UP

Required Output Format:
Please provide a JSON response ONLY (no additional text) with the following structure:

{{
    "analysis_metadata": {{
        "input_height": {height},
        "input_unit": "{unit}",
        "input_gender": "{gender}",
        "analysis_date": "{{current_date}}"
    }},
    "body_shape_analysis": "{{Shape category, e.g., Rectangle, Pear, Hourglass, etc.}}",
    "estimated_measurements": {{
        "shoulder_width": {{
            "cm": {{value}},
            "inches": {{value}}
        }},
        "chest_circumference": {{
            "cm": {{value}},
            "inches": {{value}}
        }},
        "waist_circumference": {{
            "cm": {{value}},
            "inches": {{value}}
        }},
        "hip_circumference": {{
            "cm": {{value}},
            "inches": {{value}}
        }},
        "chest_to_waist_ratio": {{
            "cm": {{value}},
            "description": "{{difference between chest and waist - used for fit determination}}"
        }},
        "sleeve_length": {{
            "cm": {{value}},
            "inches": {{value}}
        }},
        "inseam_length": {{
            "cm": {{value}},
            "inches": {{value}}
        }},
        "depth_chest": {{
            "cm": {{value}},
            "inches": {{value}},
            "note": "Measured from side profile"
        }},
        "depth_waist": {{
            "cm": {{value}},
            "inches": {{value}},
            "note": "Measured from side profile"
        }}
    }},
    "recommended_sizes": {{
        "shirt": {{
            "size": "{{S/M/L/XL/XXL}}",
            "chest_range": "{{cm}}",
            "chest_range_inches": "{{inches}}",
            "fit_type": "{{slim/regular/relaxed}}",
            "sizing_notes": "{{detailed explanation of why this size was chosen, considering all measurements}}"
        }},
        "pants_trousers": {{
            "waist_size": "{{number}}",
            "length": "{{length_in_cm}}",
            "inseam_inches": "{{inseam in inches}}",
            "fit_note": "{{regular/slim/relaxed, etc}}"
        }}
    }}
}}

IMPORTANT INSTRUCTIONS:
- Provide ONLY the JSON response, no additional text before or after
- All measurements should be as accurate as possible based on the photogrammetry analysis
- Include both cm and inches for all measurements
- CRITICAL: When calculating shirt size, use the sizing charts provided above as your primary reference
- CRITICAL: If chest circumference is 108-116 cm, size MUST be XL (not M or L)
- Be conservative in your estimates if image quality is poor, but prioritize accuracy
- Ensure the JSON is valid and properly formatted
- Do NOT calculate shirt size with only chest measurements - validate against shoulder width, sleeve length, and waist circumference
- Always include detailed sizing notes explaining your size selection based on all body measurements
- If measurements fall on size boundaries, default to the LARGER size for comfort
"""
    
    return prompt


def parse_json_response(text: str) -> Dict[str, Any]:
    """
    Parse JSON response from the model.
    Handles cases where the response might contain extra text.
    """
    try:
        # Try direct parsing first
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        try:
            # Find the first { and last }
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_str = text[start_idx:end_idx + 1]
                return json.loads(json_str)
        except:
            pass
        
        # If all else fails, return error response
        return {
            "error": "Failed to parse model response",
            "raw_response": text[:500]  # Return first 500 chars for debugging
        }


def generate_with_retry(
    prompt: str,
    front_image_part: Part,
    side_image_part: Part,
    max_retries=MAX_RETRIES
) -> Optional[str]:
    """
    Generate body size recommendation with retry logic for rate limiting.
    
    Args:
        prompt: String prompt for analysis
        front_image_part: Part object for front image
        side_image_part: Part object for side image
        max_retries: Number of retries on failure
    
    Returns:
        Analysis response string or None
    """
    for attempt in range(max_retries):
        try:
            throttle_request()
            print(f"Attempt {attempt + 1}/{max_retries}: Analyzing body measurements...")
            
            response = get_model().generate_content([
                prompt,
                front_image_part,
                side_image_part
            ])
            
            if response.candidates and response.candidates[0].content.parts:
                result = response.candidates[0].content.parts[0].text
                return result
            else:
                raise Exception("No content in response")
                
        except Exception as e:
            error_str = str(e)
            print(f"Error during attempt {attempt + 1}: {error_str}")
            
            # Check for rate limit error
            if "429" in error_str or "Resource exhausted" in error_str:
                if attempt < max_retries - 1:
                    wait_time = RETRY_DELAY * (attempt + 1)
                    print(f"⏱️ Rate limited. Waiting {wait_time} seconds before retry...")
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
                    raise HTTPException(
                        status_code=500,
                        detail=f"Analysis failed: {error_str}"
                    )
    
    return None


@router.post("/body-size-recommendation")
async def body_size_recommendation(
    front_image: UploadFile = File(..., description="Front view full-body image"),
    side_image: UploadFile = File(..., description="Side profile full-body image"),
    height: float = Form(..., description="Height of the person (default: cm)"),
    gender: str = Form(..., description="Gender (Male/Female/Other)"),
    height_unit: str = Form("cm", description="Unit of height measurement (e.g., cm, inches, m, mm, ft, etc.)"),
    include_confidence: bool = Form(True, description="Include confidence metrics in response")
) -> JSONResponse:
    """
    Analyze body measurements and provide clothing size recommendations.
    
    Args:
        front_image: Front view full-body image (required)
        side_image: Side profile full-body image (required)
        height: Height of the person (required)
        gender: Gender of the person (required)
        height_unit: Unit of height measurement - any standard unit (cm, inches, m, mm, ft, etc.) (default: cm)
        include_confidence: Whether to include confidence metrics (default: True)
    
    Returns:
        JSON response with estimated measurements and size recommendations
    
    Example:
        POST /api/gemini/body-size-recommendation
        {
            "front_image": <image_file>,
            "side_image": <image_file>,
            "height": 175,
            "gender": "Male",
            "height_unit": "cm",
            "include_confidence": true
        }
    """
    
    try:
        # Validate inputs
        if not front_image or not side_image:
            raise HTTPException(status_code=400, detail="Both front and side images are required")
        
        if height <= 0:
            raise HTTPException(status_code=400, detail="Height must be a positive number")
        
        if gender.lower() not in ["male", "female", "other"]:
            raise HTTPException(
                status_code=400,
                detail="Gender must be 'Male', 'Female', or 'Other'"
            )
        
        # Validate height_unit is not empty
        if not height_unit or not height_unit.strip():
            raise HTTPException(
                status_code=400,
                detail="Height unit cannot be empty"
            )
        
        print(f"📊 Processing body size recommendation for {gender}, Height: {height} {height_unit}")
        
        # Read image files
        front_image_data = await front_image.read()
        side_image_data = await side_image.read()
        
        if not front_image_data or not side_image_data:
            raise HTTPException(status_code=400, detail="Invalid image files")
        
        # Encode images to base64
        front_image_b64 = base64.b64decode(base64.b64encode(front_image_data))
        side_image_b64 = base64.b64decode(base64.b64encode(side_image_data))
        
        # Create image parts
        front_image_part = Part.from_data(
            data=front_image_b64,
            mime_type=mime_type_for_filename(front_image.filename)
        )
        
        side_image_part = Part.from_data(
            data=side_image_b64,
            mime_type=mime_type_for_filename(side_image.filename)
        )
        
        # Build prompt
        prompt = build_body_analysis_prompt(
            height=height,
            gender=gender.capitalize(),
            unit=height_unit.upper()
        )
        
        # Generate analysis
        response_text = generate_with_retry(
            prompt,
            front_image_part,
            side_image_part
        )
        
        if not response_text:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate body size recommendation"
            )
        
        # Parse the JSON response
        result = parse_json_response(response_text)
        
        return JSONResponse(content=result, status_code=200)
        
    except HTTPException as http_exc:
        print(f"❌ HTTP Exception: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"❌ Error: {str(e)}\n{error_trace}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


@router.post("/body-size-recommendation-batch")
async def body_size_recommendation_batch(
    front_image: UploadFile = File(..., description="Front view full-body image"),
    side_image: UploadFile = File(..., description="Side profile full-body image"),
    height: float = Form(..., description="Height of the person"),
    gender: str = Form(..., description="Gender (Male/Female/Other)"),
    height_unit: str = Form("cm", description="Unit of height measurement (e.g., cm, inches, m, mm, ft, etc.)"),
    custom_prompt: str = Form(None, description="Custom analysis prompt (optional)")
) -> JSONResponse:
    """
    Analyze body measurements with optional custom prompt.
    Useful for specialized analysis requirements.
    
    Args:
        front_image: Front view full-body image
        side_image: Side profile full-body image
        height: Height of the person
        gender: Gender of the person
        height_unit: Unit of height measurement - any standard unit (cm, inches, m, mm, ft, etc.)
        custom_prompt: Optional custom prompt to override default
    
    Returns:
        JSON response with measurements and recommendations
    """
    
    try:
        # Read image files
        front_image_data = await front_image.read()
        side_image_data = await side_image.read()
        
        if not front_image_data or not side_image_data:
            raise HTTPException(status_code=400, detail="Invalid image files")
        
        # Create image parts
        front_image_part = Part.from_data(
            data=front_image_data,
            mime_type=mime_type_for_filename(front_image.filename)
        )
        
        side_image_part = Part.from_data(
            data=side_image_data,
            mime_type=mime_type_for_filename(side_image.filename)
        )
        
        # Use custom prompt if provided, otherwise use default
        if custom_prompt:
            prompt = custom_prompt + f"\n\nInput Data: Height={height}{height_unit}, Gender={gender}"
        else:
            prompt = build_body_analysis_prompt(height, gender, height_unit)
        
        # Generate analysis
        response_text = generate_with_retry(
            prompt,
            front_image_part,
            side_image_part
        )
        
        if not response_text:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate body size recommendation"
            )
        
        # Parse response
        result = parse_json_response(response_text)
        
        return JSONResponse(content=result, status_code=200)
        
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )
