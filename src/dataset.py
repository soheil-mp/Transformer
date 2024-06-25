
# Turn of the tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import the libraries
import pickle
import numpy as np
import tensorflow as tf

# Custom class for the dataset
class PrepareDataset:

    # Constructor function
    def __init__(self, **kwargs):

        # Inherite the parent's constructor
        super().__init__(**kwargs)

        # Initialization
        self.n_sentences = 1_000     # Number of sentences in dataset
        self.train_split = 0.8        # Train split ratio
        self.val_split = 0.1          # Validation split ratio

    # Function for creating and fitting a tokenizer to given dataset
    def create_tokenizer(self, dataset):
        """
        This function creates and fits a tokenizer to given dataset.

        PARAMETERS
        ===========================
            - dataset (list): list of sentences

        RETURNS
        ===========================
            - tokenizer (tensorflow tokenizer): fitted tokenizer
        """
        
        # Initialize a tokenizer
        tokenizer = tf.keras.preprocessing.text.Tokenizer()

        # Fit the tokenizer to the dataset
        tokenizer.fit_on_texts(dataset)

        return tokenizer
    
    # Function for finding the sequence length of the dataset
    def find_sequence_length(self, dataset):
        """
        This function finds the sequence length of the dataset.

        PARAMETERS
        ===========================
            - dataset (list): list of sentences

        RETURNS
        ===========================
            - sequence_length (int): sequence length of the dataset
        """

        # Sequence length
        sequence_length = max([len(i_seq.split()) for i_seq in dataset])

        return sequence_length
    
    # Function for finding the vocabulary size
    def find_vocabulary_size(self, tokenizer, dataset):
        """
        This function finds the vocabulary size of the dataset.

        PARAMETERS
        ===========================
            - tokenizer (tensorflow tokenizer): fitted tokenizer
            - dataset (list): list of sentences

        RETURNS
        ===========================
            - vocabulary_size (int): vocabulary size of the dataset
        """

        # Fit the tokenizer to the dataset
        tokenizer.fit_on_texts(dataset)

        # Vocabulary size
        vocabulary_size = len(tokenizer.word_index) + 1

        return vocabulary_size
    
    # Function for encoding and padding the input sequences
    def encode_and_pad(self, dataset, tokenizer, sequence_length):
        """
        This function encodes and pads the input sequences.

        PARAMETERS
        ===========================
            - dataset (list): list of sentences
            - tokenizer (tensorflow tokenizer): fitted tokenizer
            - sequence_length (int): sequence length of the dataset

        RETURNS
        ===========================
            - out (tensorflow tensor): encoded and padded dataset
        """

        # Encode the dataset
        encoded_dataset = tokenizer.texts_to_sequences(dataset)

        # Pad the dataset
        padded_dataset = tf.keras.preprocessing.sequence.pad_sequences(encoded_dataset, maxlen = sequence_length, padding = "post")

        # Convert the dataset into tensor
        out = tf.convert_to_tensor(padded_dataset, dtype=tf.int64)

        return out
    
    # Function for saving the tokenizer
    def save_tokenizer(self, tokenizer, name):
        """
        This function saves the tokenizer into a pickle file.

        PARAMETERS
        ===========================
            - tokenizer (tensorflow tokenizer): fitted tokenizer
            - name (str): name of the pickle file   

        RETURNS 
        ===========================
            - None
        """
        # Open the pickle file
        with open(name + "_tokenizer.pkl", "wb") as f:

            # Dump the tokenizer
            pickle.dump(tokenizer, f, protocol = pickle.HIGHEST_PROTOCOL)

    # Call function
    def __call__(self, filename, **kwargs):
                 
        # Load the dataset (already cleaned)
        dataset_clean = pickle.load(open(filename, "rb"))

        # Sample a subset of the dataset
        dataset = dataset_clean[:self.n_sentences, :]

        # Add the START and EOS tokens to each sentence
        for i in range(dataset[:, 0].size):

            # Add the tokens to the sentences 
            dataset[i, 0] = "<START> " + dataset[i, 0] + " <EOS>"
            dataset[i, 1] = "<START> " + dataset[i, 1] + " <EOS>"

        # Shuffle the dataset
        np.random.shuffle(dataset)

        # Split the dataset into train, val, test
        train = dataset[:int(self.n_sentences * self.train_split)]
        val   = dataset[int(self.n_sentences * self.train_split)  :  int(self.n_sentences * (1 - self.val_split))]
        test  = dataset[int(self.n_sentences * (1 - self.val_split)) :]

        # Tokenization process for encoder input
        encoder_tokenizer = self.create_tokenizer(dataset[:, 0])
        encoder_sequence_length = self.find_sequence_length(dataset[:, 0])
        encoder_vocabulary_size = self.find_vocabulary_size(encoder_tokenizer, train[:, 0])

        # Tokenization process for decoder input
        decoder_tokenizer = self.create_tokenizer(dataset[:, 1])
        decoder_sequence_length = self.find_sequence_length(dataset[:, 1])
        decoder_vocabulary_size = self.find_vocabulary_size(decoder_tokenizer, train[:, 1])

        # Encode and pad the train dataset
        train_x = self.encode_and_pad(train[:, 0], encoder_tokenizer, encoder_sequence_length)
        train_y = self.encode_and_pad(train[:, 1], decoder_tokenizer, decoder_sequence_length)

        # Encode and pad the validation dataset
        val_x = self.encode_and_pad(val[:, 0], encoder_tokenizer, encoder_sequence_length)
        val_y = self.encode_and_pad(val[:, 1], decoder_tokenizer, decoder_sequence_length)

        # Save the encoder/decoder tokenizer
        self.save_tokenizer(encoder_tokenizer, "encoder")
        self.save_tokenizer(decoder_tokenizer, "decoder")

        # Save the testing dataset into a text file using savetxt
        np.savetxt("test_dataset.txt", test, fmt="%s")

        return (train_x, train_y, val_x, val_y, train, val, encoder_sequence_length, decoder_sequence_length, encoder_vocabulary_size, decoder_vocabulary_size)


# # Test out the codes
# if __name__ == "__main__":

#     # Initialize the dataset class
#     dataset = PrepareDataset()

#     # Call it 
#     train_x, train_y, val_x, val_y, train, val, encoder_sequence_length, decoder_sequence_length, encoder_vocabulary_size, decoder_vocabulary_size = dataset("./dataset/english-german-both.pkl")

#     # Report
#     print("Train X shape: ", train_x.shape)
#     print("Train Y shape: ", train_y.shape)
#     print("Val X shape: ", val_x.shape)
#     print("Val Y shape: ", val_y.shape)
#     print("Train shape: ", train.shape)
#     print("Val shape: ", val.shape)
#     print("Encoder sequence length: ", encoder_sequence_length)
#     print("Decoder sequence length: ", decoder_sequence_length)
#     print("Encoder vocabulary size: ", encoder_vocabulary_size)
#     print("Decoder vocabulary size: ", decoder_vocabulary_size)
#     print("Train X: ", train_x)
#     print("Train Y: ", train_y)
