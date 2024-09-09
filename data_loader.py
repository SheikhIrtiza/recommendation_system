import tensorflow_datasets as tfds
import tensorflow as tf

def load_data():
    ratings_dataset, _ = tfds.load(
        name='movielens/100k-ratings',
        split='train',
        with_info=True
    )
    movies_dataset, _ = tfds.load(
        name='movielens/100k-movies',
        split='train',
        with_info=True
    )
    return ratings_dataset, movies_dataset

def preprocess_data(ratings_dataset):
    ratings_dataset = ratings_dataset.map(
        lambda rating: {
            'user_id': rating['user_id'],
            'movie_id': rating['movie_id'],
            'movie_title': rating['movie_title'],
            'user_rating': rating['user_rating'],
            'timestamp': rating['timestamp']
        }
    )
    return ratings_dataset

def split_data(ratings_dataset):
    trainset_size = int(0.8 * len(ratings_dataset))
    ratings_dataset_shuffled = ratings_dataset.shuffle(buffer_size=100_000, seed=42, reshuffle_each_iteration=False)
    ratings_trainset = ratings_dataset_shuffled.take(trainset_size)
    ratings_testset = ratings_dataset_shuffled.skip(trainset_size)
    return ratings_trainset, ratings_testset