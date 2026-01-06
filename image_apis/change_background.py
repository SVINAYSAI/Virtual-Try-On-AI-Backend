from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from rembg import remove
from PIL import Image
from io import BytesIO

router = APIRouter(prefix="/api", tags=["Background Operations"])

@router.post("/change_background", summary="Remove background and composite with new background")
async def remove_and_composite(
    foreground: UploadFile = File(..., description="Image to remove background from"),
    background: UploadFile = File(..., description="Background image to apply")
):
    """
    Removes the background from the `foreground` image and composites it
    onto the `background` image. Both images are resized to match the
    dimensions of the foreground.

    Returns:
    - PNG stream of the final composited image.
    """
    try:
        # Read and process foreground
        fg_bytes = await foreground.read()
        fg_img = Image.open(BytesIO(fg_bytes)).convert("RGBA")
        fg_no_bg = remove(fg_img)

        # Read and process background
        bg_bytes = await background.read()
        bg_img = Image.open(BytesIO(bg_bytes)).convert("RGBA")
        bg_resized = bg_img.resize(fg_no_bg.size)

        # Composite images
        composite_img = Image.alpha_composite(bg_resized, fg_no_bg)

        # Prepare response buffer
        buf = BytesIO()
        composite_img.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

