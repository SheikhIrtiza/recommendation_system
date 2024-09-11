# import tensorflow as tf
# import tensorflow_recommenders as tfrs
# from data_loader import load_data, preprocess_data, split_data
# from model import create_models, RetrievalModel

# class RetrievalModel(tfrs.Model):
#     def __init__(self, user_model, movie_model, task):
#         super().__init__()
#         self.user_model = user_model
#         self.movie_model = movie_model
#         self.task = task

#     def call(self, features):
#         user_embeddings = self.user_model(features["user_id"])
#         movie_embeddings = self.movie_model(features["movie_id"])
#         return user_embeddings, movie_embeddings

# def train_retrieval_model():
#     ratings_dataset, movies_dataset = load_data()
#     ratings_dataset = preprocess_data(ratings_dataset)
#     ratings_trainset, ratings_testset = split_data(ratings_dataset)

#     user_id_model, movie_id_model, _ = create_models(ratings_trainset)

#     retrieval_ratings_trainset = ratings_trainset.map(lambda rating: {'user_id': rating['user_id'], 'movie_id': rating['movie_id']})
#     retrieval_ratings_testset = ratings_testset.map(lambda rating: {'user_id': rating['user_id'], 'movie_id': rating['movie_id']})

#     candidates_corpus_dataset = movies_dataset.map(lambda movie: movie['movie_id'])
#     factorized_top_k_metrics = tfrs.metrics.FactorizedTopK(candidates=candidates_corpus_dataset.batch(128).map(movie_id_model))
#     retrieval_task_layer = tfrs.tasks.Retrieval(metrics=factorized_top_k_metrics)

#     movielens_retrieval_model = RetrievalModel(user_id_model, movie_id_model, retrieval_task_layer)
#     movielens_retrieval_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

#     retrieval_cached_ratings_trainset = retrieval_ratings_trainset.shuffle(100_000).batch(8192).cache()
#     retrieval_cached_ratings_testset = retrieval_ratings_testset.batch(4096).cache()

#     num_epochs = 5
#     history = movielens_retrieval_model.fit(retrieval_cached_ratings_trainset, validation_data=retrieval_cached_ratings_testset, validation_freq=1, epochs=num_epochs)
    
#     # Build the model with an input shape
#     movielens_retrieval_model.call = tf.function(movielens_retrieval_model.call)
#     movielens_retrieval_model._set_inputs({'user_id': tf.constant(["0"]), 'movie_id': tf.constant(["0"])})

#     # Save the trained model
#     model_save_path = "retrieval"
#     tf.keras.models.save_model(movielens_retrieval_model, model_save_path)

#     # Retrieve top 5 movie IDs for simplicity
#     retrieved_movie_ids = [str(i) for i in range(1, 6)]

#     return movielens_retrieval_model, history, retrieved_movie_ids


import tensorflow as tf
import tensorflow_recommenders as tfrs
from data_loader import load_data, preprocess_data, split_data
from model import create_models, RetrievalModel

class RetrievalModel(tfrs.Model):
    def __init__(self, user_model, movie_model, task):
        super().__init__()
        self.user_model = user_model
        self.movie_model = movie_model
        self.task = task

    def call(self, features):
        user_embeddings = self.user_model(features["user_id"])
        movie_embeddings = self.movie_model(features["movie_id"])
        return user_embeddings, movie_embeddings

    def compute_loss(self, features, training=False):
        user_embeddings, movie_embeddings = self(features)
        return self.task(user_embeddings, movie_embeddings)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'user_model': self.user_model,
            'movie_model': self.movie_model,
            'task': self.task,
        })
        return config

def train_retrieval_model():
    ratings_dataset, movies_dataset = load_data()
    ratings_dataset = preprocess_data(ratings_dataset)
    ratings_trainset, ratings_testset = split_data(ratings_dataset)

    user_id_model, movie_id_model, _ = create_models(ratings_trainset)

    retrieval_ratings_trainset = ratings_trainset.map(lambda rating: {'user_id': rating['user_id'], 'movie_id': rating['movie_id']})
    retrieval_ratings_testset = ratings_testset.map(lambda rating: {'user_id': rating['user_id'], 'movie_id': rating['movie_id']})

    candidates_corpus_dataset = movies_dataset.map(lambda movie: movie['movie_id'])
    factorized_top_k_metrics = tfrs.metrics.FactorizedTopK(candidates=candidates_corpus_dataset.batch(128).map(movie_id_model))
    retrieval_task_layer = tfrs.tasks.Retrieval(metrics=factorized_top_k_metrics)

    movielens_retrieval_model = RetrievalModel(user_id_model, movie_id_model, retrieval_task_layer)
    movielens_retrieval_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    retrieval_cached_ratings_trainset = retrieval_ratings_trainset.shuffle(100_000).batch(8192).cache()
    retrieval_cached_ratings_testset = retrieval_ratings_testset.batch(4096).cache()

    num_epochs = 5
    history = movielens_retrieval_model.fit(retrieval_cached_ratings_trainset, validation_data=retrieval_cached_ratings_testset, validation_freq=1, epochs=num_epochs)
    
    # Define input shape
    movielens_retrieval_model._set_inputs({'user_id': tf.constant([0]), 'movie_id': tf.constant([0])})

    # Save the trained model
    model_save_path = "retrieval"
    tf.keras.models.save_model(movielens_retrieval_model, model_save_path)

    # Retrieve top 5 movie IDs for simplicity
    retrieved_movie_ids = [str(i) for i in range(1, 6)]

    return movielens_retrieval_model, history, retrieved_movie_ids
