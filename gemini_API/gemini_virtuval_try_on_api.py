from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from vertexai.preview.generative_models import GenerativeModel, Part
from io import BytesIO
from PIL import Image
import base64
import os

router = APIRouter(prefix="/api/gemini", tags=["Gemini Virtual Try-On"])

# Model will be initialized after vertexai.init() is called
model = None

def get_model():
    """Lazy load the model after Vertex AI initialization"""
    global model
    if model is None:
        model = GenerativeModel("gemini-2.5-flash-image")
    return model


# -------------------------------------------------------------
#  FULL GARMENT VIRTUAL TRY-ON ENDPOINT
# -------------------------------------------------------------
@router.post("/virtual-tryon")
async def virtual_tryon(person: UploadFile = File(...), full_garment: UploadFile = File(...)):
    try:
        # Read files
        person_bytes = await person.read()
        full_garment_bytes = await full_garment.read()

        # Mime type detection
        person_mime = f"image/{person.filename.split('.')[-1].lower()}"
        full_garment_mime = f"image/{full_garment.filename.split('.')[-1].lower()}"

        # Create Gemini parts
        person_part = Part.from_data(mime_type=person_mime, data=person_bytes)
        full_garment_part = Part.from_data(mime_type=full_garment_mime, data=full_garment_bytes)

        # Enhanced prompt for full garment
        prompt = (
            "Make the person wear the full garment realistically in a high-quality fashion photo. "
            "The model identity, face, pose, and background MUST remain the same exactly. "
            "Ensure the garment is applied with accurate texture, patterns, colors, fabric flow, "
            "stitching details, reflections, and natural fitting on the body. "
            "Do NOT change the background or the person's body shape. "
            "Generate at 19:6 aspect ratio like a vertical reel. "
            "Maintain original garment texture exactly."
        )

        response = get_model().generate_content([person_part, full_garment_part, prompt])

        # Extract returned image
        img_bytes = None
        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data.mime_type.startswith("image/"):
                data = part.inline_data.data
                img_bytes = base64.b64decode(data) if isinstance(data, str) else data
                break

        if img_bytes is None:
            raise HTTPException(status_code=500, detail="Model returned no image.")

        # Save output image with unique naming
        os.makedirs("outputs", exist_ok=True)
        output_path = "outputs/virtual_tryon_result.png"
        with open(output_path, "wb") as f:
            f.write(img_bytes)

        return FileResponse(output_path, media_type="image/png", filename="virtual_tryon_result.png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



