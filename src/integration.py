import numpy as np
from numpy.linalg import norm


def cosine_similarity(a, b):
    """
    Computes cosine similarity between a 1D array and a 2D array.
    """
    return np.dot(a, b.T) / (norm(a) * norm(b, axis=1))


def find_most_similar_image(uploaded_features, dataset_features_flat):
    """
    Finds the most similar image by cosine similarity.
    """
    similarities = cosine_similarity(uploaded_features, dataset_features_flat)
    most_similar_index = np.argmax(similarities)
    max_similarity = similarities[most_similar_index]
    return most_similar_index, max_similarity
