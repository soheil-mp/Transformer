
# Import the libraries
import pickle
import tensorflow as tf
from transformer import TransformerModel
from hyperparameters import *

# Dataset parameters
enc_seq_length = 7      # Encoder sequence length
dec_seq_length = 9      # Decoder sequence length
enc_vocab_size = 786    # Encoder vocabulary size
dec_vocab_size = 1031   # Decoder vocabulary size

# Initialize the model
inferencing_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v, d_model, d_ff, n, 0)

# Custom class for inference
class Inference(tf.Module):

    # Constructor function
    def __init__(self, inferencing_model, **kwargs):
        
        # Inherite the parent's constructor
        super().__init__(**kwargs)

        # Initialize the model
        self.transformer = inferencing_model

    # Function for loading the tokenizer
    def load_tokenizer(self, name):

        # Load the tokenizer from the specified file
        with open(name, 'rb') as handle: return pickle.load(handle)

    # Call function
    def __call__(self, sentence):

        # Append START and EOS tokens to the input sentence
        sentence[0] = "<START> " + sentence[0] + " <EOS>"

        # Load tokenizers for encoder and decoder
        enc_tokenizer = self.load_tokenizer('enc_tokenizer.pkl')
        dec_tokenizer = self.load_tokenizer('dec_tokenizer.pkl')

        # Encoder input; tokenizing, padding and converting to tensor
        encoder_input = enc_tokenizer.texts_to_sequences(sentence)
        encoder_input = tf.keras.preprocessing.sequence.pad_sequences(encoder_input, maxlen=enc_seq_length, padding='post')
        encoder_input = tf.convert_to_tensor(encoder_input, dtype=tf.int64)

        # Start of the output sequence is with the <START> token
        output_start = dec_tokenizer.texts_to_sequences(["<START>"])            # Convert to integers
        output_start = tf.convert_to_tensor(output_start[0], dtype=tf.int64)    # Convert to tensor

        # End of the output sequence is with the <EOS> token (for breaking the loop)
        output_end = dec_tokenizer.texts_to_sequences(["<EOS>"])               # Convert to integers
        output_end = tf.convert_to_tensor(output_end[0], dtype=tf.int64)      # Convert to tensor

        # Output array for storing the predicted tokens (with dynamic size)
        decoder_output = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        decoder_output = decoder_output.write(0, output_start)

        # Loop over the decoder sequence length
        for i in range(dec_seq_length):

            # Prediction
            prediction = self.transformer(encoder_input,tf. transpose(decoder_output.stack()), training=False)
            
            # Select the last predicted token
            prediction = prediction[:, -1, :]

            # Select the prediction with the highest score
            predicted_id = tf.argmax(prediction, axis=-1)
            predicted_id = predicted_id[0][tf.newaxis]

            # Write the selected prediction to the output array at the next available index
            decoder_output = decoder_output.write(i + 1, predicted_id)

            # Break if an <EOS> token is predicted
            if predicted_id == output_end: break

        # Transpose the output array and convert to numpy array
        output = tf.transpose(decoder_output.stack())[0]
        output = output.numpy()

        # Initialize an empty list for storing the output string
        output_str = []

        ### Decode the predicted tokens into an output string

        # Loop over the output array
        for i in range(output.shape[0]):

            # Select the token at the current index
            key = output[i]

            # Append the token to the output string
            output_str.append(dec_tokenizer.index_word[key])

        return output_str



# TESTING
sentence = ['im thirsty']                                  # Sentence to translate
inferencing_model.load_weights('weights/wghts5.ckpt')      # Load the trained model's weights at the specified epoch
translator = Inference(inferencing_model)                  # Create a new instance of the 'Translate' class
print(translator(sentence))                                # Translate the input sentence

