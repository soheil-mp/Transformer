
# Turn off the tensorflow loggings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import the libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# POsition embedding layer
class PositionEmbeddingLayer(tf.keras.layers.Layer):

    # Constructor function
    def __init__(self, seq_length, vocab_size, output_dim, **kwargs):

        # Inherite the parent's constructor
        super().__init__(**kwargs)

        # Word embedding layer
        self.word_embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=output_dim)

        # Position embedding layer
        self.position_embedding_layer = tf.keras.layers.Embedding(input_dim=seq_length, output_dim=output_dim)

    # Call function
    def call(self, inputs):

        # Initialize the positions
        position_indices = tf.range(start=0, limit=tf.shape(inputs)[-1])

        # Feed words and positions to embedding layer
        embedded_words = self.word_embedding_layer(inputs)
        embedded_positions = self.position_embedding_layer(position_indices)

        # Sum up the embeddings
        out = embedded_words + embedded_positions

        return out
    

# Positional embedding layer class with initializing the weights with sine/cosine function
class PositionEmbeddingLayerWithFixedWeights(tf.keras.layers.Layer):

    # Constructor function
    def __init__(self, seq_length, vocab_size, output_dim, **kwargs):

        # Inherite parent's constructor
        super().__init__(**kwargs)

        # Initialize the word / position embedding matrix (for initializing the weights with them)
        word_embedding_matrix = self.get_position_encoding(vocab_size, output_dim)
        position_embedding_matrix = self.get_position_encoding(seq_length, output_dim)

        # Initialize the word / position embedding layer
        self.word_embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, 
                                                              output_dim=output_dim,
                                                              weights=[word_embedding_matrix],
                                                              trainable=False)
        self.position_embedding_layer = tf.keras.layers.Embedding(input_dim=seq_length,
                                                                  output_dim=output_dim,
                                                                  weights=[position_embedding_matrix],
                                                                  trainable=False
                                                                  )
        
    # Get position encoding (sine/cosine)
    def get_position_encoding(self, seq_len, d, n=10_000):

        # Initialize the positional matrix
        P = np.zeros(shape=(seq_len, d))

        # Loop over the range of sequence length
        for k in range(seq_len):

            # Loop over the index from 0 to d/2
            for i in np.arange(0, int(d/2)):

                # Denominator
                denominator = np.power(n, 2*i/d)

                # Sine
                P[k, 2*i] = np.sin(k/denominator)

                # Cosine
                P[k, 2*i+1] = np.cos(k/denominator)

        return P
    
    # Call function
    def call(self, inputs):

        # Initialize the position indices
        position_indices = tf.range(start=0, limit=tf.shape(inputs)[-1])

        # Feed the word and position data into the embedding layer
        embedded_words = self.word_embedding_layer(inputs)
        embedded_positions = self.position_embedding_layer(position_indices)

        # Sum up the embedding layers
        out = embedded_words + embedded_positions

        return out