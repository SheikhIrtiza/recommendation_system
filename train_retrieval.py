import tensorflow as tf
import tensorflow_recommenders as tfrs
from data_loader import load_data, preprocess_data, split_data
from model import create_models, RetrievalModel

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
    
    # Retrieve top 5 movie IDs for simplicity
    retrieved_movie_ids = [str(i) for i in range(1, 6)]


    return movielens_retrieval_model, history, retrieved_movie_ids
