import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Load MobileNetV2 without the classification head
model = MobileNetV2(weights="imagenet", include_top=False, pooling='avg')

def extract_features(image_path):
    """
    Given a path to an image, returns a feature vector extracted by MobileNetV2.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))  # MobileNetV2 expects 224x224
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        features = model.predict(image, verbose=0)
        return features.flatten()
    
    except Exception as e:
        print(f"[ERROR] Feature extraction failed for {image_path}: {e}")
        raise e
