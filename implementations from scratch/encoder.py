
# Turn off the tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import the libraries
import numpy as np
import tensorflow as tf
from multihead_attention import MultiHeadAttentionLayer
from positional_encoding import PositionEmbeddingLayer, PositionEmbeddingLayerWithFixedWeights

# Custom layer for Add & Norm layer
class AddNormalizationLayer(tf.keras.layers.Layer):

    # Constructor function
    def __init__(self, **kwargs):

        # Inherite the parent's constructor
        super().__init__(**kwargs)

        # Initialize the layer normalization layer
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    # Call function
    def call(self, x, sublayer_x):

        # Add the sublayer input and output together
        add = x + sublayer_x

        # Apply layer normalization
        out = self.layer_norm(add)

        return out
    
# Custom layer for Feed-Forward layer
class FeedForwardLayer(tf.keras.layers.Layer):

    # Constructor function
    def __init__(self, d_ff, d_model, **kwargs):

        # Inherite the parent's constructor
        super().__init__(**kwargs)

        # Initialize the dense layers
        self.fully_connected_1 = tf.keras.layers.Dense(units=d_ff)
        self.fully_connected_2 = tf.keras.layers.Dense(units=d_model)

        # Initialize the activation function
        self.activation = tf.keras.layers.ReLU()

    # Call function
    def call(self, x):

        # Feed the data
        x = self.fully_connected_1(x)
        x = self.activation(x)
        x = self.fully_connected_2(x)

        return x
    
# Custom layer for Transformer Encoder
class EncoderLayer(tf.keras.layers.Layer):

    # Constructor function
    def __init__(self, sequence_length, h, d_k, d_v, d_model, d_ff, rate, **kwargs):

        # Inherite the parent's constructor
        super().__init__(**kwargs)

        # Initialization (to use in rest of the class)
        self.sequence_length = sequence_length 
        self.d_model = d_model

        # Attention layer
        self.multihead_attention = MultiHeadAttentionLayer(h, d_k, d_v, d_model)

        # Feed-forward layer
        self.feed_forward = FeedForwardLayer(d_ff, d_model)

        # Add & Norm layer
        self.add_norm_1 = AddNormalizationLayer()
        self.add_norm_2 = AddNormalizationLayer()

        # Dropout layer
        self.dropout_1 = tf.keras.layers.Dropout(rate)
        self.dropout_2 = tf.keras.layers.Dropout(rate)

        # Build the model
        self.build(input_shape=[None, sequence_length, d_model])

    # Build function
    def build_graph(self):

        # Construct the model
        input_layer = tf.keras.layers.Input(shape=(self.sequence_length, self.d_model))
        return tf.keras.Model(inputs=[input_layer], outputs=self.call(input_layer, None, True))

    # Call function
    def call(self, x, padding_mask, training):

        # Feed the data
        multihead_output   = self.multihead_attention(x, x, x, padding_mask)
        multihead_output   = self.dropout_1(multihead_output, training=training)
        addnorm_output     = self.add_norm_1(x, multihead_output)
        feedforward_output = self.feed_forward(addnorm_output)
        feedforward_output = self.dropout_2(feedforward_output, training=training)
        addnorm_output     = self.add_norm_2(addnorm_output, feedforward_output)

        return addnorm_output

# Custom layer for the full model
class Encoder(tf.keras.layers.Layer):

    # Constructor function
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate, **kwargs):

        # Inherite the parent's constructor
        super().__init__(**kwargs)

        # Positional encoding layer
        self.positional_encoding = PositionEmbeddingLayerWithFixedWeights(sequence_length, vocab_size, d_model)

        # Dropout layer
        self.dropout = tf.keras.layers.Dropout(rate)

        # Encoder layers (for N times)
        self.encoder_layer = [EncoderLayer(sequence_length, h, d_k, d_v, d_model, d_ff, rate) for _ in range(n)]

    # Call function
    def call(self, input_sentence, padding_mask, training):

        # Feed the data
        x = self.positional_encoding(input_sentence)
        x = self.dropout(x, training=training) 
        for i_index, i_layer in enumerate(self.encoder_layer):
            x = i_layer(x, padding_mask, training)

        return x
    
# Initialization
h = 8                     # Number of self-attention heads
d_k = 64                  # Dimension of the key and query vectors
d_v = 64                  # Dimension of the value vectors
d_ff = 2048               # Dimension of the inner feed-forward layer
d_model = 512             # Dimension of the mode syb-layer' output
n = 6                     # Number of encoder layers
batch_size = 64           # Batch size
dropout_rate = 0.1        # Dropout rate
enc_vocab_size = 8192     # Encoder vocabulary size
input_seq_length = 64     # Maximum length of the input sequence

# Test model output
input_seq = np.random.random((batch_size, input_seq_length))
encoder = Encoder(enc_vocab_size, input_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)
output_encoder = encoder(input_seq, None, True)
print("Output of the encoder: \n", output_encoder)

# Model summary
encoder_layer = EncoderLayer(input_seq_length, h, d_k, d_v, d_model, d_ff, dropout_rate)
print("Model summary: ", encoder_layer.build_graph().summary())