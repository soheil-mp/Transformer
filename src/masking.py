
# Turn off the tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import the libraries
import numpy as np
import tensorflow as tf

# Function for masking the paddings
def padding_mask(inputs):

    # Create mask (i.e. marks zero paddings by a 1 and 0 elsewhere)
    mask = tf.math.equal(inputs, 0)
    mask = tf.cast(mask, tf.float32)

    return mask

# Function for lookahead mask
def lookahead_mask(shape):

    # Create mask (i.e. marks future enteries by a 1 and 0 elsewhere)
    mask = 1 - tf.linalg.band_part(tf.ones((shape, shape)), num_lower=-1, num_upper=0)

    return mask


# # Test the padding mask
# inputs = np.array([1, 2, 3, 4, 0, 0, 0])
# masked_inputs = padding_mask(inputs)
# print("Inputs: ", inputs)
# print("Masked inputs: ", masked_inputs)

# # Test the lookahead
# masked_lookahead = lookahead_mask(shape=5)
# print("Masked lookahead of shape 5: ", masked_lookahead)