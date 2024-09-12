from train_retrieval import train_retrieval_model
from train_ranking import train_ranking_model
import numpy as np

def main():
    user_id = input("Enter user ID: ")

    # Run the retrieval stage
    retrieval_model, history, retrieved_movie_titles = train_retrieval_model()

    # Use retrieval_model and history if needed
    # For example, print them
    print(f"Retrieval Model: {retrieval_model}")
    print(f"History: {history}")

    # Run the ranking stage with retrieved movie IDs
    ranking_model = train_ranking_model(retrieved_movie_titles)

    # Get recommendations for a specific user
    test_ratings = {}
    test_movie_titles = ["M*A*S*H (1970)", "Dances with Wolves (1990)", "Speed (1994)"]
    for movie_title in test_movie_titles:
        test_ratings[movie_title] = ranking_model({
            "user_id": np.array([user_id]),  # Use the input user_id here
            "movie_title": np.array([movie_title])
        })

    print("Ratings:")
    for id, score in sorted(test_ratings.items(), key=lambda x: x[1], reverse=True):
        print(f"{id}: {score}")

if __name__ == "__main__":
    main()
