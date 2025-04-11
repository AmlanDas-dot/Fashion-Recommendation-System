import joblib
# Train KNN model
knn = NearestNeighbors(n_neighbors=5, algorithm='auto')
knn.fit(features)

# Save the KNN model
joblib.dump(knn, 'knn_model.pkl')
