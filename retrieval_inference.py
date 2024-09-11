# import tensorflow as tf
# import numpy as np

# # Load the ranking model from the saved_model directory
# ranking_model = tf.saved_model.load('retrieval')


import tensorflow as tf
import numpy as np

# Load the ranking model from the saved_model directory
ranking_model = tf.saved_model.load('retrieval')

# Check if the model loaded successfully
try:
    # Attempt to call the model with a dummy input
    dummy_input = np.random.rand(1, 10).astype(np.float32)
    output = ranking_model(dummy_input)
    print("Model loaded successfully and produced an output.")
except Exception as e:
    print(f"Failed to load the model: {e}")
