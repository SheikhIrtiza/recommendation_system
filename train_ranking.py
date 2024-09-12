import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from model import MovielensRankingModel

def train_ranking_model(retrieved_movie_titles):
    ratings = tfds.load("movielens/100k-ratings", split="train")
    ratings = ratings.map(lambda x: {"movie_title": x["movie_title"], "user_id": x["user_id"], "user_rating": x["user_rating"]})

    tf.random.set_seed(42)
    shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
    train = shuffled.take(80_000)
    test = shuffled.skip(80_000).take(20_000)

    movie_titles = ratings.batch(1_000_000).map(lambda x: x["movie_title"])
    user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))

    model = MovielensRankingModel(unique_user_ids, unique_movie_titles)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    cached_train = train.shuffle(100_000).batch(8192).cache()
    cached_test = test.batch(4096).cache()

    model.fit(cached_train, epochs=3)
    model.evaluate(cached_test, return_dict=True)

    tf.saved_model.save(model, "ranking_model")


    return model
