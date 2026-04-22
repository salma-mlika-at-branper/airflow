import logging
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# This load will happen synchronously on module load
from model_loader import predict, generate_opinion
logger.info("Models loaded and ready")

# Initialize app
app = FastAPI(title="FastAPI Sentiment Analysis App")

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic model for request validation
class SentimentRequest(BaseModel):
    text: str

@app.get("/")
async def serve_index():
    """Serves the main HTML page."""
    return FileResponse("static/index.html")

@app.post("/predict")
async def predict_endpoint(request: SentimentRequest):
    """Predicts sentiment for the given text."""
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
        
    try:
        result = predict(request.text)
        return result
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/generate")
async def generate_endpoint(request: SentimentRequest):
    """Predicts sentiment and generates a short analytical opinion using T5."""
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
        
    try:
        result = generate_opinion(request.text)
        return result
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
