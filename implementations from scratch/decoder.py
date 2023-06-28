
# Turn off the tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import the libraries
import numpy as np
import tensorflow as tf
from multihead_attention import MultiHeadAttentionLayer
from positional_encoding import PositionEmbeddingLayerWithFixedWeights
from encoder import AddNormalizationLayer, FeedForwardLayer

# Custom layer for decoder layer
class DecoderLayer(tf.keras.layers.Layer):

    # Constructor function
    def __init__(self, sequence_length, h, d_k, d_v, d_model, d_ff, rate, **kwargs):

        # Inherite parent's constructor
        super().__init__(**kwargs)

        # Multi-head attention layer
        self.multihead_attention_1 = MultiHeadAttentionLayer(h, d_k, d_v, d_model)
        self.multihead_attention_2 = MultiHeadAttentionLayer(h, d_k, d_v, d_model)

        # Feed-forward layer
        self.feed_forward = FeedForwardLayer(d_ff, d_model)

        # Add & Norm layer
        self.add_norm_1 = AddNormalizationLayer()
        self.add_norm_2 = AddNormalizationLayer()
        self.add_norm_3 = AddNormalizationLayer()

        # Dropout layer
        self.dropout_1 = tf.keras.layers.Dropout(rate)
        self.dropout_2 = tf.keras.layers.Dropout(rate)
        self.dropout_3 = tf.keras.layers.Dropout(rate)

        # Initialization
        self.sequence_length = sequence_length
        self.d_model = d_model

        # Build the model
        self.build(input_shape=[None, sequence_length, d_model])

    # Build function
    def build_graph(self):

        # Construct the model
        input_layer = tf.keras.layers.Input(shape=(self.sequence_length, self.d_model))
        return tf.keras.Model(inputs=[input_layer], outputs=self.call(input_layer, input_layer, None, None, True))

    # Call function
    def call(self, x, encoder_output, lookahead_mask, padding_mask, training):

        # Feed the data
        multihead_output_1 = self.multihead_attention_1(x, x, x, lookahead_mask)
        multihead_output_1 = self.dropout_1(multihead_output_1, training=training)
        addnorm_output_1 = self.add_norm_1(x, multihead_output_1)

        multihead_output_2 = self.multihead_attention_2(addnorm_output_1, encoder_output, encoder_output, padding_mask)
        multihead_output_2 = self.dropout_2(multihead_output_2, training=training)
        addnorm_output_2 = self.add_norm_2(addnorm_output_1, multihead_output_2)

        feedforward_output = self.feed_forward(addnorm_output_2)
        feedforward_output = self.dropout_3(feedforward_output, training=training)
        addnorm_output_3 = self.add_norm_3(addnorm_output_2, feedforward_output)

        return addnorm_output_3
    
# Custom layer for the full decoder model
class Decoder(tf.keras.layers.Layer):

    # Constructor function
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate, **kwargs):

        # Inherite the parent's constructor
        super().__init__(**kwargs)

        # Positional encoding layer
        self.pos_encoding = PositionEmbeddingLayerWithFixedWeights(sequence_length, vocab_size, d_model)

        # Dropout layer
        self.dropout = tf.keras.layers.Dropout(rate)

        # Decoder layers (for N times)
        self.decoder_layer = [DecoderLayer(sequence_length, h, d_k, d_v, d_model, d_ff, rate) for _ in range(n)]

    # Call function
    def call(self, output_target, encoder_output, lookahead_mask, padding_mask, training):

        # Feed the data
        x = self.pos_encoding(output_target)
        x = self.dropout(x, training=training)
        for i_index, i_layer in enumerate(self.decoder_layer):
            x = i_layer(x, encoder_output, lookahead_mask, padding_mask, training)

        return x
    

# Hyperparameters
h = 8                  # Number of heads
d_k = 64               # Dimension of the key and query vectors
d_v = 64               # Dimension of the value vectors
d_ff = 2048            # Dimension of the feed-forward layer
d_model = 512          # Dimension of the model sub-layer' output
n = 6                  # Number of encoder layers
batch_size = 64        # Batch size
dropout_rate = 0.1     # Dropout rate
dec_vocab_size = 8000  # Decoder vocabulary size
input_seq_length = 40  # Input sequence length

# Test model output
input_seq = np.random.random((batch_size, input_seq_length))
encoder_output = np.random.random((batch_size, input_seq_length, d_model))
decoder = Decoder(dec_vocab_size, input_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)
output_target = decoder(input_seq, encoder_output, None, True)
print("Decoder Output: ", output_target)

# Model summary
decoder_layer = DecoderLayer(input_seq_length, h, d_k, d_v, d_model, d_ff, dropout_rate)
decoder_layer.build_graph().summary()

