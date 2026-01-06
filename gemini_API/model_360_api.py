from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
import base64
import os
import json
import time
from threading import Lock
from vertexai.preview.generative_models import GenerativeModel, Part
from io import BytesIO
import zipfile

router = APIRouter(prefix="/api/gemini", tags=["360 Model Generation"])

# Model will be initialized after vertexai.init() is called
gen_model = None

# Rate limiting configuration
REQUEST_LOCK = Lock()
LAST_REQUEST_TIME = 0
MIN_REQUEST_INTERVAL = 10  # Minimum 5 seconds between requests
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
#   API: MULTI-ANGLE MODEL GENERATION (4 VIEWS - CONSISTENT MODEL)
#   Direct Download - Returns ZIP with all images
# =====================================================================
@router.get("/generate-model-360")
async def generate_model_360(
    gender: str = Query("female", description="male / female"),
    region: str = Query("european", description="indian / european / korean / african / latino / arabic / american"),
    age_group: str = Query("young adult", description="child / teen / young adult / adult"),
    background: str = Query("studio", description="studio / white / outdoor / beach / city / indoor / minimal"),
    outfit_type: str = Query("modern wear", description="casual / modern wear / streetwear / formal / winter / summer"),
    custom_prompt: str = Query("", description="Additional custom instructions for model generation (optional)"),
    return_format: str = Query("zip", description="zip / json - zip returns downloadable file, json returns image data info")
):
    """
    Generate 4 different angle images of the SAME model with SAME garment.
    Returns: Front view, Back view, Detailed shot, Full-length shot
    
    This API:
    1. First generates a base model with specific facial features
    2. Then uses that base image to generate different angles while maintaining consistency
    3. Returns a downloadable ZIP file with all images OR JSON with image data
    
    Parameters:
    - gender: male / female
    - region: indian / european / korean / african / latino / arabic / american
    - age_group: child / teen / young adult / adult
    - background: studio / white / outdoor / beach / city / indoor / minimal
    - outfit_type: casual / modern wear / streetwear / formal / winter / summer
    - custom_prompt: Additional text to include in generation (optional)
    - return_format: 'zip' for downloadable file (default), 'json' for JSON response
    """
    try:
        # STEP 1: Generate base model with front view and distinctive features
        print("STEP 1: Generating base model...")
        
        base_model_prompt = (
            f"Create a high-quality photorealistic human fashion model with VERY DISTINCTIVE and MEMORABLE facial features. "
            f"Gender: {gender}. Ethnicity: {region}. Age: {age_group}. "
            f"Outfit: {outfit_type}. Background: {background}. "
            f"CRITICAL FEATURES TO LOCK:\n"
            f"- Face: Unique facial structure, distinctive eye shape, specific nose shape, memorable lips, unique eyebrows\n"
            f"- Body: Specific body type, shoulder width, height proportions\n"
            f"- Outfit: Exact same garment, exact same colors, exact same patterns\n"
            f"Front facing pose. Standing straight, looking directly at camera. Full body visible from head to toe. "
            f"Sharp, clear facial details. Professional fashion model presentation. 19:6 aspect ratio."
        )
        
        if custom_prompt:
            base_model_prompt += f"\nAdditional requirements: {custom_prompt}"
        
        base_img_bytes = generate_with_retry(base_model_prompt)
        
        if base_img_bytes is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate base model. Please try again."
            )
        
        print(f"✓ Base model generated successfully")
        
        # STEP 2: Create image part for base model
        base_image_part = Part.from_data(mime_type="image/png", data=base_img_bytes)
        
        # STEP 3: Generate 4 different angles using the base model as reference
        print("STEP 2: Generating different angles from base model...")
        
        angles_config = {
            "front_view": {
                "description": "Front facing model view",
                "pose_instruction": "Show the model in FRONT FACING POSE, looking directly at camera, standing upright.",
                "details": "Full body visible from head to toe. Face centered, sharp and clear. Clean frontal fashion presentation."
            },
            "back_view": {
                "description": "Back view of the same model",
                "pose_instruction": "Show the EXACT SAME MODEL from the BACK, with back to camera, head slightly turned.",
                "details": "Full body visible. Show back of garment clearly. Same body proportions, same outfit color."
            },
            "detailed_shot": {
                "description": "Close-up detailed shot",
                "pose_instruction": "Show a MID-SHOT of the SAME MODEL focusing on face and upper body/shoulders.",
                "details": "Face sharp and clear showing distinctive features. Shoulders and garment details visible. Professional close-up photography."
            },
            "full_shot": {
                "description": "Full-length dynamic pose",
                "pose_instruction": "Show the SAME MODEL in a NATURAL DYNAMIC POSE (slightly relaxed, one leg forward, natural arm position).",
                "details": "Full body visible from head to toe. Natural confident stance. Same outfit, same model, professional presentation."
            }
        }
        
        results = {}
        failed_angles = []
        generated_images = {
            "base_model": base_img_bytes
        }
        
        angle_names = list(angles_config.keys())
        
        for idx, angle_name in enumerate(angle_names):
            angle_info = angles_config[angle_name]
            
            # Create prompt that references the base model image
            angle_prompt = (
                f"You have a reference image of a fashion model. Your task is to generate a NEW image showing "
                f"THE EXACT SAME PERSON in a different pose/angle.\n\n"
                f"CRITICAL REQUIREMENTS:\n"
                f"1. SAME FACE: Keep the same facial features - same eye shape, nose, mouth, face structure, eyebrows\n"
                f"2. SAME BODY: Keep the same body proportions and body type\n"
                f"3. SAME OUTFIT: Show the exact same garment, same colors, same patterns\n"
                f"4. SAME BACKGROUND: Keep the same background type and lighting\n"
                f"5. NEW POSE: {angle_info['pose_instruction']}\n\n"
                f"Additional details: {angle_info['details']}\n"
                f"Generate in 19:6 aspect ratio. Photorealistic quality. Professional fashion photography."
            )
            
            print(f"Generating {angle_name} ({idx + 1}/4)...")
            
            try:
                # Send base image + text prompt to Gemini
                response = generate_with_retry([base_image_part, angle_prompt])
                
                if response is None:
                    failed_angles.append({
                        "angle": angle_name,
                        "error": "No image data returned"
                    })
                    print(f"✗ {angle_name} failed: No image returned")
                    continue
                
                generated_images[angle_name] = response
                
                results[angle_name] = {
                    "status": "success",
                    "size": len(response),
                    "description": angle_info["description"]
                }
                
                print(f"✓ {angle_name} generated successfully")
                
                # Add delay between requests (except after last one)
                if idx < len(angle_names) - 1:
                    print(f"Waiting 3 seconds before next request...")
                    time.sleep(3)
                    
            except HTTPException as http_err:
                # Re-raise HTTP exceptions (rate limits, etc.)
                raise http_err
            except Exception as e:
                failed_angles.append({
                    "angle": angle_name,
                    "error": str(e)
                })
                print(f"✗ {angle_name} failed: {str(e)}")
        
        # Check if we got at least some images
        if not results:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate any angle images. Errors: {failed_angles}"
            )
        
        results["base_model"] = {
            "status": "success",
            "size": len(base_img_bytes),
            "description": "Initial base model used for reference"
        }
        
        # STEP 4: Return based on format
        if return_format.lower() == "zip":
            # Create ZIP file with all images
            print("Creating ZIP file...")
            zip_buffer = BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add all generated images
                for image_name, image_data in generated_images.items():
                    zip_file.writestr(f"{image_name}.png", image_data)
                
                # Add metadata JSON
                metadata = {
                    "model_type": "360_view_consistent",
                    "gender": gender,
                    "region": region,
                    "age_group": age_group,
                    "outfit_type": outfit_type,
                    "background": background,
                    "custom_prompt": custom_prompt,
                    "generation_method": "Two-step: Base model generation + Angle transformation",
                    "generated_images": results,
                    "failed_generations": failed_angles if failed_angles else None,
                    "consistency_note": "All 4 angles show the SAME model (same face, same body, same garment) in different poses",
                    "total_generated": len([r for r in results.values() if r.get("status") == "success"]),
                    "total_failed": len(failed_angles),
                    "images": {
                        "base_model": "Reference model - front view",
                        "front_view": "Model from front angle",
                        "back_view": "Model from back angle",
                        "detailed_shot": "Close-up of face and upper body",
                        "full_shot": "Full body in dynamic pose"
                    }
                }
                zip_file.writestr("metadata.json", json.dumps(metadata, indent=2))
            
            zip_buffer.seek(0)
            
            return StreamingResponse(
                iter([zip_buffer.getvalue()]),
                media_type="application/zip",
                headers={"Content-Disposition": "attachment; filename=model_360_views.zip"}
            )
        
        else:  # return_format == "json"
            # Return JSON with image information
            metadata = {
                "model_type": "360_view_consistent",
                "gender": gender,
                "region": region,
                "age_group": age_group,
                "outfit_type": outfit_type,
                "background": background,
                "custom_prompt": custom_prompt,
                "generation_method": "Two-step: Base model generation + Angle transformation",
                "generated_images": results,
                "failed_generations": failed_angles if failed_angles else None,
                "consistency_note": "All 4 angles show the SAME model (same face, same body, same garment) in different poses",
                "total_generated": len([r for r in results.values() if r.get("status") == "success"]),
                "total_failed": len(failed_angles),
                "note": "To get the actual image files, use return_format=zip parameter"
            }
            
            return {
                "status": "partial_success" if failed_angles else "success",
                "message": f"Generated {len([r for r in results.values() if r.get('status') == 'success'])}/5 images (4 angles + 1 base model)" if failed_angles else "360-view model generation completed successfully!",
                "results": results,
                "metadata": metadata,
                "failed_angles": failed_angles if failed_angles else None,
                "notes": "All images show the SAME person from different angles. Face, body, and outfit are consistent across all views."
            }
    
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
