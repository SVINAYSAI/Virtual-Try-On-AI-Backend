from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from PIL import Image
import pilgram
import io

router = APIRouter(prefix="/api/filter", tags=["Image Filters"])


# ---------------------------------------------
# OPENCV CUSTOM FILTERS
# ---------------------------------------------

def morning_light(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.add(hsv[:,:,2], 40)
    hsv[:,:,1] = cv2.add(hsv[:,:,1], 20)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def afternoon_filter(img):
    b, g, r = cv2.split(img)
    r = cv2.add(r, 40)
    g = cv2.add(g, 10)
    return cv2.merge((b, g, r))

def evening_golden(img):
    b, g, r = cv2.split(img)
    r = cv2.add(r, 60)
    g = cv2.add(g, 20)
    return cv2.merge((b, g, r))

def night_filter(img):
    b, g, r = cv2.split(img)
    b = cv2.add(b, 40)
    g = cv2.add(g, 10)
    r = cv2.subtract(r, 20)
    return cv2.merge((b, g, r))

def beauty_smooth(img):
    return cv2.bilateralFilter(img, 20, 75, 75)

def high_contrast(img):
    return cv2.convertScaleAbs(img, alpha=1.5, beta=20)

def cartoon(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(blur, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 5)
    color = cv2.bilateralFilter(img, 15, 250, 250)
    return cv2.bitwise_and(color, color, mask=edges)


# ---------------------------------------------
# PILGRAM FILTER WRAPPER
# ---------------------------------------------

def pilgram_filter(img, filter_func):
    pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    filtered = filter_func(pil_image)
    return cv2.cvtColor(np.array(filtered), cv2.COLOR_RGB2BGR)


# ---------------------------------------------
# FILTER DICTIONARY (OpenCV + Pilgram)
# ---------------------------------------------

FILTERS = {
    # --- Custom Filters ---
    "morning": morning_light,
    "afternoon": afternoon_filter,
    "evening": evening_golden,
    "night": night_filter,
    "beauty": beauty_smooth,
    "contrast": high_contrast,
    "cartoon": cartoon,

    # --- Pilgram Instagram-like filters ---
    "aden": lambda img: pilgram_filter(img, pilgram.aden),
    "brooklyn": lambda img: pilgram_filter(img, pilgram.brooklyn),
    "clarendon": lambda img: pilgram_filter(img, pilgram.clarendon),
    "gingham": lambda img: pilgram_filter(img, pilgram.gingham),
    "hudson": lambda img: pilgram_filter(img, pilgram.hudson),
    "inkwell": lambda img: pilgram_filter(img, pilgram.inkwell),
    "kelvin": lambda img: pilgram_filter(img, pilgram.kelvin),
    "lark": lambda img: pilgram_filter(img, pilgram.lark),
    "lofi": lambda img: pilgram_filter(img, pilgram.lofi),
    "mayfair": lambda img: pilgram_filter(img, pilgram.mayfair),
    "moon": lambda img: pilgram_filter(img, pilgram.moon),
    "nashville": lambda img: pilgram_filter(img, pilgram.nashville),
    "reyes": lambda img: pilgram_filter(img, pilgram.reyes),
    "slumber": lambda img: pilgram_filter(img, pilgram.slumber),
    "valencia": lambda img: pilgram_filter(img, pilgram.valencia),
    "walden": lambda img: pilgram_filter(img, pilgram.walden),
    "willow": lambda img: pilgram_filter(img, pilgram.willow)
}

# ---------------------------------------------
# API ENDPOINTS
# ---------------------------------------------

@router.get("/list")
async def list_filters():
    """Get list of all available filters"""
    return {
        "total_filters": len(FILTERS),
        "custom_filters": ["morning", "afternoon", "evening", "night", "beauty", "contrast", "cartoon"],
        "instagram_filters": [
            "aden", "brooklyn", "clarendon", "gingham", "hudson", "inkwell", 
            "kelvin", "lark", "lofi", "mayfair", "moon", "nashville", 
            "reyes", "slumber", "valencia", "walden", "willow"
        ],
        "all_filters": list(FILTERS.keys())
    }


@router.post("/apply")
async def apply_filter(
    file: UploadFile = File(...),
    filter_name: str = Query(..., description="Filter name to apply")
):
    """Apply a filter to an uploaded image"""
    try:
        # Validate filter name
        if filter_name not in FILTERS:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid filter: {filter_name}. Available filters: {list(FILTERS.keys())}"
            )
        
        # Read uploaded image
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Apply selected filter
        filtered_img = FILTERS[filter_name](img)
        
        # Encode back to bytes
        success, buffer = cv2.imencode('.png', filtered_img)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode filtered image")
        
        # Return as streaming response
        io_buf = io.BytesIO(buffer)
        return StreamingResponse(io_buf, media_type="image/png")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
