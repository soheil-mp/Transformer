
# Import the libraries
import numpy as np
import tensorflow as tf
from encoder import Encoder
from decoder import Decoder

# Custom class for transformer mode
class TransformerModel(tf.keras.Model):

    # Constructor function
    def __init__(self, enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate, **kwargs):

        # Inherite the parent's constructor
        super().__init__(**kwargs)

        # Encoder model
        self.encoder = Encoder(enc_vocab_size, enc_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate)

        # Decoder model
        self.decoder = Decoder(dec_vocab_size, dec_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate)

        # Dense layer
        self.last_layer = tf.keras.layers.Dense(dec_vocab_size)

    # Function for masking the padding
    def padding_mask(self, inputs):

        # Create mask (i.e. marks zero paddings by a 1 and 0 elsewhere)
        mask = tf.math.equal(inputs, 0)
        mask = tf.cast(mask, tf.float32)

        # Make the shape broadcastable for the attention weights
        mask = mask[:, tf.newaxis, tf.newaxis, :]

        return mask
    
    # Function for masking the lookahead
    def lookahead_mask(self, shape):

        # Create mask (i.e. marks future words by a 1 and 0 elsewhere)
        mask = 1 - tf.linalg.band_part(tf.ones((shape, shape)), -1, 0)

        return mask
    
    # Function for calling the model
    def call(self, encoder_input, decoder_input, training):

        # Mask the paddings for input data 
        enc_padding_mask = self.padding_mask(encoder_input)
        dec_in_padding_mask = self.padding_mask(decoder_input)

        # Mask the lookahead
        dec_in_lookahead_mask = self.lookahead_mask(decoder_input.shape[1])
        dec_in_lookahead_mask = tf.maximum(dec_in_padding_mask, dec_in_lookahead_mask)

        # Feed to encoder
        encoder_output = self.encoder(encoder_input, enc_padding_mask, training)

        # Feed to decoder
        decoder_output = self.decoder(decoder_input, encoder_output, dec_in_lookahead_mask, enc_padding_mask, training)

        # Feed to the last layer
        model_output = self.last_layer(decoder_output)



        return model_output


# # Hyperparaneters
# enc_vocab_size = 20   # Vocabulary size for the encoder
# dec_vocab_size = 20   # Vocabulary size for the decoder
# enc_seq_length = 5    # Maximum length of the input sequence
# dec_seq_length = 5    # Maximum length of the target sequence
# h = 8                 # Number of self-attention heads
# d_k = 64              # Dimensionality of the linearly projected queries and keys
# d_v = 64              # Dimensionality of the linearly projected values
# d_ff = 2048           # Dimensionality of the inner fully connected layer
# d_model = 512         # Dimensionality of the model sub-layers' outputs
# n = 6                 # Number of layers in the encoder stack
# dropout_rate = 0.1    # Frequency of dropping the input units in the dropout layers

# # Create model
# training_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)