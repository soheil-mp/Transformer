
# Turn off the tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import the libraries
import tensorflow as tf
import numpy as np

# Scaled dot product attention class
class ScaledDotProductAttentionLayer(tf.keras.layers.Layer):

    # Constructor function
    def __init__(self, **kwargs):

        # Inherite the parent's constructor
        super().__init__(**kwargs)


    # Call function
    def call(self, queries, keys, values, d_k, mask=None):

        # Attention score
        attention_score = tf.matmul(queries, keys, transpose_b=True) / tf.math.sqrt(tf.cast(d_k, tf.float32))

        # Apply mask
        if mask is not None:
            attention_score += (-1e9 * mask)

        # Apply softmax
        weights = tf.keras.backend.softmax(attention_score)

        # Calculate the weighted sum 
        out = tf.matmul(weights, values)

        return out
    

# Multi-head attention class
class MultiHeadAttentionLayer(tf.keras.layers.Layer):

    # Constructor function
    def __init__(self, h, d_k, d_v, d_model, **kwargs):

        # Inherite the parent's constructor
        super().__init__(**kwargs)

        # Initialize the scaled dot product attention layer
        self.attention = ScaledDotProductAttentionLayer()

        # Initialization
        self.heads = h            # Number of attention heads
        self.d_k = d_k            # Dimension of the key vector (and also the query vector)
        self.d_v = d_v            # Dimension of the value vector
        self.d_model = d_model    # Dimension of the model

        # Initialize dense layer for learned projection matrix for queries, keys, values, and model
        self.W_q = tf.keras.layers.Dense(units=d_k)
        self.W_k = tf.keras.layers.Dense(units=d_k)
        self.W_v = tf.keras.layers.Dense(units=d_v)
        self.W_o = tf.keras.layers.Dense(units=d_model)


    # Function for reshaping the tensor 
    def reshape_tensor(self, x, heads, flag):

        # If flag is on
        # Used when recieving the linearly projected queries, keys, or values as input
        # Final shape should be: (batch_size, heads, seq_length, -1)
        if flag:

            # Reshape the tensor
            x = tf.reshape(x, shape=(tf.shape(x)[0], tf.shape(x)[1], heads, -1))

            # Transpose the tensor
            x = tf.transpose(x, perm=(0, 2, 1, 3))

        # If flag is off
        # Use after the data feeded into the multi head attention layer
        # Final shape should be: (batch_size, seq_length, d_k)
        else:

            # Transpose
            x = tf.transpose(x, perm=(0, 2, 1, 3))

            # Reshape
            x = tf.reshape(x, shape=(tf.shape(x)[0], tf.shape(x)[1], self.d_k))

        return x
    

    # Call function
    def call(self, queries, keys, values, mask=None):

        # Reshape queries, keys, values to be able to compute all heads in parallel
        queries_reshaped = self.reshape_tensor(self.W_q(queries), self.heads, True)
        keys_reshaped = self.reshape_tensor(self.W_k(keys), self.heads, True)
        values_reshaped = self.reshape_tensor(self.W_v(values), self.heads, True)
        
        # Compute multi-head attention
        output_reshaped = self.attention(queries_reshaped, keys_reshaped, values_reshaped, self.d_k, mask)

        # Rearrange the output into concatenated form
        output = self.reshape_tensor(output_reshaped, self.heads, False)

        # Apply the linear projection to the output
        output = self.W_o(output)

        return output
