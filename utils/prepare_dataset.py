import os
import numpy as np
import pandas as pd
import joblib
from models.feature_extractor import extract_features

def prepare_dataset(
    styles_csv_path='data/styles.csv',
    image_dir='data/images',
    category_filter='Apparel',
    subcategory_filter=None,
    features_path='saved_models/features.npy',
    paths_path='saved_models/image_paths.pkl'
):
    # Load the styles.csv
    styles_df = pd.read_csv(styles_csv_path, on_bad_lines='skip')

    # Filter by main category
    filtered_df = styles_df[styles_df['masterCategory'] == category_filter]

    # Optional: Filter by subcategory (e.g., Topwear, Bottomwear)
    if subcategory_filter:
        filtered_df = filtered_df[filtered_df['subCategory'] == subcategory_filter]

    features = []
    image_paths = []

    for img_id in filtered_df['id']:
        img_path = os.path.join(image_dir, f"{img_id}.jpg")
        if os.path.exists(img_path):
            try:
                feat = extract_features(img_path)
                features.append(feat)
                image_paths.append(img_path)
            except Exception as e:
                print(f"Skipping {img_id}.jpg due to error: {e}")
                continue

    features = np.array(features)
    np.save(features_path, features)
    joblib.dump(image_paths, paths_path)

    print(f"Done! {len(features)} features saved.")
    return features, image_paths
