import tensorflow as tf
import tensorflow_recommenders as tfrs

def create_models(ratings_trainset):
    user_id_lookup_layer = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
    user_id_lookup_layer.adapt(ratings_trainset.map(lambda x: x['user_id']))
    user_id_embedding_layer = tf.keras.layers.Embedding(input_dim=user_id_lookup_layer.vocab_size(), output_dim=32)
    user_id_model = tf.keras.Sequential([user_id_lookup_layer, user_id_embedding_layer])

    movie_id_lookup_layer = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
    movie_id_lookup_layer.adapt(ratings_trainset.map(lambda x: x['movie_id']))
    movie_id_embedding_layer = tf.keras.layers.Embedding(input_dim=movie_id_lookup_layer.vocab_size(), output_dim=32)
    movie_id_model = tf.keras.Sequential([movie_id_lookup_layer, movie_id_embedding_layer])

    movie_title_vectorization_layer = tf.keras.layers.experimental.preprocessing.TextVectorization()
    movie_title_vectorization_layer.adapt(ratings_trainset.map(lambda rating: rating['movie_title']))
    movie_title_embedding_layer = tf.keras.layers.Embedding(input_dim=len(movie_title_vectorization_layer.get_vocabulary()), output_dim=32, mask_zero=True)
    movie_title_model = tf.keras.Sequential([movie_title_vectorization_layer, movie_title_embedding_layer, tf.keras.layers.GlobalAveragePooling1D()])

    return user_id_model, movie_id_model, movie_title_model

# class RetrievalModel(tfrs.models.Model):
#     def __init__(self, query_model, candidate_model, retrieval_task_layer):
#         super().__init__()
#         self.query_model = query_model
#         self.candidate_model = candidate_model
#         self.retrieval_task_layer = retrieval_task_layer

#     def compute_loss(self, features, training=False):
#         query_embeddings = self.query_model(features['user_id'])
#         positive_candidate_embeddings = self.candidate_model(features["movie_id"])
#         loss = self.retrieval_task_layer(query_embeddings, positive_candidate_embeddings)
#         return loss

class RetrievalModel(tfrs.models.Model):
    def __init__(self, query_model, candidate_model, retrieval_task_layer):
        super().__init__()
        self.query_model = query_model
        self.candidate_model = candidate_model
        self.retrieval_task_layer = retrieval_task_layer

    def call(self, features):
        query_embeddings = self.query_model(features['user_id'])
        candidate_embeddings = self.candidate_model(features['movie_title'])
        return query_embeddings, candidate_embeddings

    def compute_loss(self, features, training=False):
        query_embeddings, positive_candidate_embeddings = self(features)
        loss = self.retrieval_task_layer(query_embeddings, positive_candidate_embeddings)
        return loss


class RankingModel(tf.keras.Model):
    def __init__(self, unique_user_ids, unique_movie_titles):
        super().__init__()
        embedding_dimension = 32
        self.user_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])
        self.movie_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_movie_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
        ])
        self.ratings = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1)
        ])

    def call(self, inputs):
        user_id, movie_title = inputs
        user_embedding = self.user_embeddings(user_id)
        movie_embedding = self.movie_embeddings(movie_title)
        return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))

class MovielensRankingModel(tfrs.models.Model):
    def __init__(self, unique_user_ids, unique_movie_titles):
        super().__init__()
        self.ranking_model = RankingModel(unique_user_ids, unique_movie_titles)
        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, features):
        return self.ranking_model((features["user_id"], features["movie_title"]))

    def compute_loss(self, features, training=False):
        labels = features.pop("user_rating")
        rating_predictions = self(features)
        return self.task(labels=labels, predictions=rating_predictions)