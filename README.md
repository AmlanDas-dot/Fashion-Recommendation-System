AI-Powered Fashion Recommendation System:

Overview-

The AI-Powered Fashion Recommendation System is a web-based application that recommends visually similar fashion items and outfit combinations using image-based similarity analysis. The project combines Deep Learning, Machine Learning, and Generative AI techniques to create an interactive fashion recommendation experience.

The system uses a pre-trained MobileNetV2 model to extract visual features from fashion images and a K-Nearest Neighbors (KNN) model to identify similar products. Users can select items from different fashion categories and receive recommendations based on visual similarity.

Additionally, the project includes an AI Fashion Assistant powered by Llama 2 through Ollama, enabling users to ask fashion-related questions and receive styling suggestions.


 Objectives:

* Develop an intelligent fashion recommendation system using image similarity.
* Generate outfit suggestions based on selected fashion items.
* Apply Deep Learning techniques for image feature extraction.
* Use Machine Learning algorithms for similarity-based recommendations.
* Integrate a conversational AI assistant for fashion guidance.
* Provide an easy-to-use web interface for users.



 Key Features:

Fashion Recommendation Engine-

Users can select fashion items from categories such as tops, bottoms, shoes, and accessories. The system recommends visually similar products based on image features extracted from the dataset.

 Deep Feature Extraction:

A pre-trained MobileNetV2 Convolutional Neural Network (CNN) is used to extract meaningful visual features from fashion images. Each image is converted into a numerical feature vector that captures characteristics such as style, texture, color, and design patterns.

Similarity Search Using KNN:

The extracted feature vectors are used to train a K-Nearest Neighbors (KNN) model. During recommendation, the system identifies the nearest feature vectors and retrieves similar fashion items from the dataset.

 Outfit Suggestions-

Based on the selected products, the system displays recommended fashion items that can be used to explore possible outfit combinations.

AI Fashion Assistant-

The project integrates Llama 2 using Ollama to provide a chatbot capable of answering fashion-related questions, offering styling suggestions, and assisting users in making fashion choices.

 Interactive User Interface-

The application is built using Streamlit, providing a simple and user-friendly interface for generating recommendations and interacting with the AI assistant.

 Testing and Validation-

Testing modules are included to validate feature extraction and recommendation functionality, helping ensure the reliability of the recommendation pipeline.


 System Architecture:

## System Architecture

```text
Dataset
   ↓
MobileNetV2 Feature Extraction
   ↓
Feature Embeddings
   ↓
KNN Similarity Search
   ↓
Fashion Recommendations
   ↓
Streamlit Web Interface
   ↓
AI Fashion Assistant (Llama 2 via Ollama)
```



 Technologies Used:

Programming Language-

* Python

 Frontend-

* Streamlit

Deep Learning-

* TensorFlow
* Keras
* MobileNetV2

Machine Learning:

* Scikit-learn (K-Nearest Neighbors)

 Data Processing:

* NumPy
* Pandas
* Joblib

Image Processing:

* Pillow (PIL)

 Generative AI:

* Ollama
* Llama 2



 How the Recommendation System Works:

1. Fashion images are loaded from the dataset.
2. MobileNetV2 extracts visual features from each image.
3. Feature vectors are stored for future use.
4. A KNN model is trained on the extracted features.
5. Users select fashion items through the Streamlit interface.
6. Features are extracted from the selected items.
7. The trained KNN model finds the nearest neighbors in the feature space.
8. Similar fashion items are retrieved and displayed as recommendations.
9. Users can interact with the AI Fashion Assistant for additional fashion advice.



 Installation-

## Installation

```bash
# Clone the repository
git clone <repository-url>

# Navigate to the project directory
cd Fashion-Recommendation-System

# Install required dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```




 Future Enhancements:

* Improved outfit matching across categories
* User profile and preference management
* Enhanced recommendation accuracy
* Larger fashion dataset support
* Cloud deployment for wider accessibility
* Integration with online fashion platforms



 Contributors:

This project was developed as a collaborative academic project focusing on Computer Vision, Machine Learning, Deep Learning, and Generative AI applications in the fashion domain.
