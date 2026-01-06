from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from rembg import remove
from PIL import Image
import io

router = APIRouter()

@router.post("/api/remove-bg")
async def remove_bg(file: UploadFile = File(...)):
    try:
        # Read image bytes
        image_bytes = await file.read()
        input_image = Image.open(io.BytesIO(image_bytes))
        # Remove background
        output_image = remove(input_image)
        # Save to bytes
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return StreamingResponse(img_byte_arr, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
