import tensorflow as tf
import numpy as np

# Load the ranking model from the saved_model directory
ranking_model = tf.saved_model.load('ranking_model')

# Sample user ID for which to get recommendations
user_id = '42'

# Pre-retrieved movie IDs from the retrieval model
retrieved_movie_titles = ["M*A*S*H (1970)", "Dances with Wolves (1990)", "Speed (1994)"]  # These would come from your retrieval phase

# Step 1: Rank the pre-retrieved movie IDs using the ranking model
def rank_movies(user_id, movie_titles, ranking_model):
    # Assuming ranking_model has a method like 'serve' for inference
    ranking_func = ranking_model.signatures['serving_default']
    test_ratings = {}

    for movie_title in movie_titles:
        # Prepare input for ranking; format based on the expected model input
        input_dict = {
            "user_id": tf.constant([user_id]),      # User ID for ranking input
            "movie_title": tf.constant([movie_title])     # Movie ID from retrieved movies
        }
        # Get rating score from the ranking model
        # Access the output using the correct key from the model
        rating_score = ranking_func(**input_dict)['output_1'].numpy()[0][0]  # Replace 'output_1' with actual output key
        test_ratings[movie_title] = rating_score

    # Sort movies by ranking scores (descending order)
    ranked_movie_titles = sorted(test_ratings.items(), key=lambda x: x[1], reverse=True)
    
    # Return both movie IDs and their corresponding scores
    return ranked_movie_titles

# Step 2: Get ranked movie IDs along with their scores
ranked_movie_titles_with_scores = rank_movies(user_id, retrieved_movie_titles, ranking_model)

# Print the ranked movie IDs and their scores
print(f"Final ranked recommendations for user {user_id}:")
for movie_title, score in ranked_movie_titles_with_scores[:5]:  # Top 5 results
    print(f"Movie ID: {movie_title}, Score: {score}")
