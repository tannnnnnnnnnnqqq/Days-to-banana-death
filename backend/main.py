from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import sys
from pathlib import Path

# Add model directory to path
sys.path.append(str(Path(__file__).parent.parent / "model"))
from predict import predictor

app = FastAPI(title="Banana Death Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model on startup
MODEL_PATH = Path(__file__).parent.parent / "model" / "saved_models" / "banana_model.keras"

@app.on_event("startup")
async def load_model():
    """Load the trained model when the API starts"""
    if MODEL_PATH.exists():
        predictor.load_model(str(MODEL_PATH))
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"⚠️ Warning: Model not found at {MODEL_PATH}")
        print("   The API will use dummy predictions until you train a model.")

@app.get("/")
async def root():
    return {
        "message": "Banana Death Predictor API is running! 🍌",
        "model_loaded": predictor.model is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None
    }

@app.post("/predict")
async def predict_banana_death(file: UploadFile = File(...)):
    """
    Upload a banana image and get prediction
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and open image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get prediction from model
        result = predictor.predict(image)
        result["status"] = "success"
        result["model_used"] = predictor.model is not None
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)