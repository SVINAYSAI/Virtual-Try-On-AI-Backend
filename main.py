from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio

# Initialize Vertex AI before importing routers
from config import initialize_vertexai
initialize_vertexai()

from image_apis.remove_bg_api import router as remove_bg_router
from image_apis.change_background import router as change_bg_router
from filter_API.filter_api import router as filter_router
from gemini_API.gemini_virtuval_try_on_api import router as virtual_tryon_router
from gemini_API.gemini_two_piece_tryon_api import router as two_piece_tryon_router
from gemini_API.human_model_api import router as model_generator_router
from gemini_API.model_360_api import router as model_360_router
from gemini_API.transform_image_api import router as transform_image_router
from gemini_API.accessory_tryon_api import router as single_tryon_router
from gemini_API.Occasion_Based_Styling import router as styling_suggestions_router
from gemini_API.makeup_tryon_api import router as makeup_tryon_router
from gemini_API.body_size_recommendation_api import router as body_size_router


# from background_crud import router as background_crud_router

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Register routers
app.include_router(remove_bg_router)
app.include_router(change_bg_router)
app.include_router(filter_router)
app.include_router(virtual_tryon_router)
app.include_router(two_piece_tryon_router)
app.include_router(model_generator_router)
app.include_router(model_360_router)
app.include_router(transform_image_router)
app.include_router(single_tryon_router)
app.include_router(styling_suggestions_router)
app.include_router(makeup_tryon_router)
app.include_router(body_size_router)

@app.get("/")
async def root():
    return {
        "message": "Mesikas Backend API",
        "version": "1.0.0",
        "endpoints": {
            "remove_bg": "/api/remove-bg",
            "change_background": "/api/change_background",
            "filter_list": "/api/filter/list",
            "filter_apply": "/api/filter/apply?filter_name={filter}",
            "virtual_tryon_full": "/api/gemini/virtual-tryon",
            "virtual_tryon_two_piece": "/api/gemini/virtual-tryon-two-piece",
            "generate_model": "/api/gemini/generate-model",
            "generate_model_360": "/api/gemini/generate-model-360",
            "transform_image": "/api/gemini/transform-image",
            "single_accessory_tryon": "/api/gemini/single-accessory-tryon",
            "styling_suggestions": "/api/styling/suggestions",
            "makeup_tryon": "/api/gemini/makeup-tryon",
            "body_size_recommendation": "/api/gemini/body-size-recommendation",
            "body_size_recommendation_batch": "/api/gemini/body-size-recommendation-batch"
        }
    }

if __name__ == "__main__":
    
    # Run the server (no reload here; use CLI for reload)
    uvicorn.run(app, host="0.0.0.0", port=8002)
    # For hot reload, run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload