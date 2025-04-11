import numpy as np
from models.feature_extractor import extract_features
from utils.outfit_generator import get_similar_items
import joblib

def test_extract_features():
    test_img = 'data/images/test_image.jpg'
    feat = extract_features(test_img)
    assert feat.shape == (2048,), "Feature extraction failed."
    print("✅ Feature extraction test passed.")

def test_knn():
    knn = joblib.load('saved_models/knn_model.pkl')
    test_feat = np.random.rand(1, 2048)
    distances, indices = knn.kneighbors(test_feat)
    assert len(indices[0]) == 5, "KNN model test failed."
    print("✅ KNN model test passed.")

if __name__ == "__main__":
    test_extract_features()
    test_knn()
