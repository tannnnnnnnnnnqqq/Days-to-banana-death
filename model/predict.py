import numpy as np
from PIL import Image
import tensorflow as tf
from typing import Tuple

class BananaPredictor:
    def __init__(self, model_path: str = None):
        """
        Initialize the banana predictor
        """
        self.model = None
        self.model_path = model_path
        
        # Updated to match dataset categories
        self.ripeness_map = {
            0: {"stage": "overripe", "days": 1},
            1: {"stage": "ripe", "days": 3},
            2: {"stage": "rotten", "days": 0},
            3: {"stage": "unripe", "days": 7}
        }
        
        # Class names (alphabetical order - how TensorFlow loads them)
        self.class_names = ['overripe', 'ripe', 'rotten', 'unripe']
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load trained model"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"✅ Model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for model input
        """
        # Resize to model input size
        image = image.resize((224, 224))
        
        # Convert to array
        img_array = np.array(image)
        
        # Normalize pixel values to 0-1
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image: Image.Image) -> dict:
        """
        Predict banana ripeness and days until death
        """
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        if self.model:
            # Use trained model
            predictions = self.model.predict(processed_image, verbose=0)
            ripeness_class = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][ripeness_class])
            
            # Get class name
            class_name = self.class_names[ripeness_class]
        else:
            # Dummy prediction (for testing without trained model)
            print("⚠️ No model loaded - using dummy prediction")
            ripeness_class = 1  # ripe
            confidence = 0.85
            class_name = 'ripe'
        
        # Get ripeness info
        ripeness_info = self.ripeness_map.get(ripeness_class, self.ripeness_map[1])
        
        return {
            "days_until_death": ripeness_info["days"],
            "ripeness_stage": ripeness_info["stage"],
            "confidence": round(confidence, 2),
            "ripeness_class": ripeness_class
        }

# Create global predictor instance
predictor = BananaPredictor()