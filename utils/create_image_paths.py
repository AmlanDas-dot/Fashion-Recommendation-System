import os
import pandas as pd
import joblib

def create_image_paths(
    styles_csv_path='data/styles.csv',
    image_dir='data/images',
    category_filter='Apparel',
    subcategory_filter=None,
    output_path='saved_models/image_paths.pkl'
):
    # Load the styles.csv
    styles_df = pd.read_csv(styles_csv_path, on_bad_lines='skip')

    # Filter based on category
    filtered_df = styles_df[styles_df['masterCategory'] == category_filter]
    if subcategory_filter:
        filtered_df = filtered_df[filtered_df['subCategory'] == subcategory_filter]

    image_paths = []
    for img_id in filtered_df['id']:
        img_path = os.path.join(image_dir, f"{img_id}.jpg")
        if os.path.exists(img_path):
            image_paths.append(img_path)

    joblib.dump(image_paths, output_path)
    print(f" Saved {len(image_paths)} image paths to {output_path}")

    return image_paths
