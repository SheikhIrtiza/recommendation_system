# import tensorflow as tf
# import tensorflow_recommenders as tfrs
# from data_loader import load_data, preprocess_data, split_data
# from model import create_models, RetrievalModel

# def train_retrieval_model():
#     ratings_dataset, movies_dataset = load_data()
#     ratings_dataset = preprocess_data(ratings_dataset)
#     ratings_trainset, ratings_testset = split_data(ratings_dataset)

#     user_id_model, movie_title_model, _ = create_models(ratings_trainset)

#     retrieval_ratings_trainset = ratings_trainset.map(lambda rating: {'user_id': rating['user_id'], 'movie_title': rating['movie_title']})
#     retrieval_ratings_testset = ratings_testset.map(lambda rating: {'user_id': rating['user_id'], 'movie_title': rating['movie_title']})

#     candidates_corpus_dataset = movies_dataset.map(lambda movie: movie['movie_title'])
#     factorized_top_k_metrics = tfrs.metrics.FactorizedTopK(candidates=candidates_corpus_dataset.batch(128).map(movie_title_model))
#     retrieval_task_layer = tfrs.tasks.Retrieval(metrics=factorized_top_k_metrics)

#     movielens_retrieval_model = RetrievalModel(user_id_model, movie_title_model, retrieval_task_layer)
#     movielens_retrieval_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

#     retrieval_cached_ratings_trainset = retrieval_ratings_trainset.shuffle(100_000).batch(8192).cache()
#     retrieval_cached_ratings_testset = retrieval_ratings_testset.batch(4096).cache()

#     num_epochs = 5
#     history = movielens_retrieval_model.fit(retrieval_cached_ratings_trainset, validation_data=retrieval_cached_ratings_testset, validation_freq=1, epochs=num_epochs)
    
#     # Retrieve top 5 movie IDs for simplicity
#     retrieved_movie_titles = [str(i) for i in range(1, 4)]


#     return movielens_retrieval_model, history, retrieved_movie_titles







#with def compile
# import tensorflow as tfimport tensorflow as tf
import tensorflow_recommenders as tfrs
from data_loader import load_data, preprocess_data, split_data
from model import create_models, RetrievalModel
import tensorflow as tf

def train_retrieval_model():
    # Load and preprocess the data
    ratings_dataset, movies_dataset = load_data()
    ratings_dataset = preprocess_data(ratings_dataset)
    ratings_trainset, ratings_testset = split_data(ratings_dataset)

    # Create the user and movie models
    user_id_model, movie_title_model, _ = create_models(ratings_trainset)

    # Prepare training and testing data for retrieval
    retrieval_ratings_trainset = ratings_trainset.map(lambda rating: {'user_id': rating['user_id'], 'movie_title': rating['movie_title']})
    retrieval_ratings_testset = ratings_testset.map(lambda rating: {'user_id': rating['user_id'], 'movie_title': rating['movie_title']})

    # Create the candidates corpus dataset
    candidates_corpus_dataset = movies_dataset.map(lambda movie: movie['movie_title'])

    # Instantiate FactorizedTopK metrics directly
    factorized_top_k_metrics = tfrs.metrics.FactorizedTopK(
        candidates=candidates_corpus_dataset.batch(128).map(movie_title_model)
    )

    # Create the retrieval task layer with the metric instance
    retrieval_task_layer = tfrs.tasks.Retrieval(metrics=factorized_top_k_metrics)

    # Initialize the retrieval model
    movielens_retrieval_model = RetrievalModel(user_id_model, movie_title_model, retrieval_task_layer)
    movielens_retrieval_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    # Cache the datasets for performance
    retrieval_cached_ratings_trainset = retrieval_ratings_trainset.shuffle(100_000).batch(8192).cache()
    retrieval_cached_ratings_testset = retrieval_ratings_testset.batch(4096).cache()

    # Train the model
    num_epochs = 5
    history = movielens_retrieval_model.fit(
        retrieval_cached_ratings_trainset, 
        validation_data=retrieval_cached_ratings_testset, 
        validation_freq=1, 
        epochs=num_epochs
    )

    # Save the model
    try:
        tf.saved_model.save(movielens_retrieval_model, "retrieval_model")
        print("Retrieval model saved successfully.")
    except Exception as e:
        print(f"Error saving the retrieval model: {e}")
        # Save weights as an alternative approach
        movielens_retrieval_model.save_weights("retrieval_model_weights.h5")
        print("Model weights saved successfully as a workaround.")

    # Retrieve top 5 movie titles for simplicity
    retrieved_movie_titles = [str(i) for i in range(1, 6)]

    return movielens_retrieval_model, history, retrieved_movie_titles

# Execute the training function
train_retrieval_model()

# import tensorflow_recommenders as tfrs
# from data_loader import load_data, preprocess_data, split_data
# from model import create_models, RetrievalModel

# def train_retrieval_model():
#     ratings_dataset, movies_dataset = load_data()
#     ratings_dataset = preprocess_data(ratings_dataset)
#     ratings_trainset, ratings_testset = split_data(ratings_dataset)

#     user_id_model, movie_title_model, _ = create_models(ratings_trainset)

#     retrieval_ratings_trainset = ratings_trainset.map(lambda rating: {'user_id': rating['user_id'], 'movie_title': rating['movie_title']})
#     retrieval_ratings_testset = ratings_testset.map(lambda rating: {'user_id': rating['user_id'], 'movie_title': rating['movie_title']})

#     candidates_corpus_dataset = movies_dataset.map(lambda movie: movie['movie_title'])
#     factorized_top_k_metrics = tfrs.metrics.FactorizedTopK(candidates=candidates_corpus_dataset.batch(128).map(movie_title_model))
#     retrieval_task_layer = tfrs.tasks.Retrieval(metrics=factorized_top_k_metrics)

#     movielens_retrieval_model = RetrievalModel(user_id_model, movie_title_model, retrieval_task_layer)
#     movielens_retrieval_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

#     retrieval_cached_ratings_trainset = retrieval_ratings_trainset.shuffle(100_000).batch(8192).cache()
#     retrieval_cached_ratings_testset = retrieval_ratings_testset.batch(4096).cache()

#     num_epochs = 5
#     history = movielens_retrieval_model.fit(retrieval_cached_ratings_trainset, validation_data=retrieval_cached_ratings_testset, validation_freq=1, epochs=num_epochs)
    
#     # Specify input shape and save the retrieval model
#     movielens_retrieval_model.call = tf.function(movielens_retrieval_model.call)
#     movielens_retrieval_model.call.get_concrete_function({
#         'user_id': tf.TensorSpec(shape=(None,), dtype=tf.string),
#         'movie_title': tf.TensorSpec(shape=(None,), dtype=tf.string)
#     })
#     tf.saved_model.save(movielens_retrieval_model, "retrieval_model_export")
    
#     # Retrieve top 5 movie IDs for simplicity
#     retrieved_movie_titles = [str(i) for i in range(1, 6)]

#     return movielens_retrieval_model, history, retrieved_movie_titles



# import tensorflow as tf
# import tensorflow_recommenders as tfrs
# from data_loader import load_data, preprocess_data, split_data
# from model import create_models, RetrievalModel

# # Define the CustomLayer class
# class CustomLayer(tf.keras.layers.Layer):
#     def __init__(self, units=32):
#         super(CustomLayer, self).__init__()
#         self.units = units
#         self.state = tf.Variable(initial_value=tf.zeros((units,)), trainable=False)

#     def call(self, inputs):
#         return inputs + self.state

#     def get_config(self):
#         config = super(CustomLayer, self).get_config()
#         config.update({"units": self.units})
#         return config

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)

# def train_retrieval_model():
#     ratings_dataset, movies_dataset = load_data()
#     ratings_dataset = preprocess_data(ratings_dataset)
#     ratings_trainset, ratings_testset = split_data(ratings_dataset)

#     user_id_model, movie_id_model, _ = create_models(ratings_trainset)

#     # Integrate the CustomLayer into the user_id_model
#     user_id_model.add(CustomLayer(units=32))

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
    
#     # Specify input shape and save the retrieval model
#     @tf.function(input_signature=[{
#         'user_id': tf.TensorSpec(shape=(None,), dtype=tf.string),
#         'movie_id': tf.TensorSpec(shape=(None,), dtype=tf.string)
#     }])
#     def serve_model(inputs):
#         return movielens_retrieval_model(inputs)

#     tf.saved_model.save(movielens_retrieval_model, "retrieval_model_export", signatures={'serving_default': serve_model})

#     # Retrieve top 5 movie IDs for simplicity
#     retrieved_movie_ids = [str(i) for i in range(1, 6)]

#     return movielens_retrieval_model, history, retrieved_movie_ids
