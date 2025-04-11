import numpy as np
import joblib
from models.feature_extractor import extract_features

def get_similar_items(user_selected_img_paths, knn_model_path='saved_models/knn_model.pkl', features_path='saved_models/features.npy', paths_path='saved_models/image_paths.pkl'):
    features = np.load(features_path)
    image_paths = joblib.load(paths_path)
    knn = joblib.load(knn_model_path)

    user_features = [extract_features(path) for path in user_selected_img_paths]
    user_features = np.array(user_features)
    distances, indices = knn.kneighbors(user_features)

    similar_items = []
    for idx_list in indices:
        similar = [image_paths[i] for i in idx_list]
        similar_items.append(similar)
    return similar_items
