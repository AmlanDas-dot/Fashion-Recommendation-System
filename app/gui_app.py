import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from PIL import Image
from utils.outfit_generator import get_similar_items

st.title('Fashion Recommendation System')

categories = ['Tops', 'Bottoms', 'Shoes', 'Accessories']
items = {
    'Tops': ['top1.jpg', 'top2.jpg'],
    'Bottoms': ['bottom1.jpg', 'bottom2.jpg'],
    'Shoes': ['shoe1.jpg', 'shoe2.jpg'],
    'Accessories': ['accessory1.jpg', 'accessory2.jpg']
}

user_selection = []

for category in categories:
    st.subheader(f'Select 5 items for {category}')
    selected = st.multiselect(f'{category} Items', items[category])
    user_selection.extend([f'data/images/{item}' for item in selected])

if st.button('Generate Outfit'):
    recommendations = get_similar_items(user_selection)
    st.subheader('Recommended Outfit Combinations')
    for outfit in zip(*recommendations):
        cols = st.columns(len(outfit))
        for idx, img_path in enumerate(outfit):
            with cols[idx]:
                st.image(Image.open(img_path), use_column_width=True)
