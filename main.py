from train_retrieval import train_retrieval_model
from train_ranking import train_ranking_model
import numpy as np

def main():

    user_id = input("Enter user ID: ")

    # Run the retrieval stage
    retrieval_model, history, retrieved_movie_ids = train_retrieval_model()

    # Run the ranking stage with retrieved movie IDs
    ranking_model = train_ranking_model(retrieved_movie_ids)

    # Get recommendations for a specific user
    test_ratings = {}
    test_movie_ids = ["47", "579", "101", "2908", "1571"]
    for movie_id in test_movie_ids:
        test_ratings[movie_id] = ranking_model({
            "user_id": np.array(["49"]),
            "movie_id": np.array([movie_id])
        })

    print("Ratings:")
    for id, score in sorted(test_ratings.items(), key=lambda x: x[1], reverse=True):
        print(f"{id}: {score}")

if __name__ == "__main__":
    main()
