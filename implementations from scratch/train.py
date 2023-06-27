
# Import the libraries
import pickle
import numpy as np
import tensorflow as tf

# Class for preprating the dataset
class PrepareDataset:

    # Constructor function
    def __init__(self, **kwargs):

        # Inherite the parent's constructor
        super().__init__(**kwargs)

        # Initialization
        self.n_sentences = 10_000     # Number of sentences in the dataset
        self.train_split = 0.9        # Ratio of the training data

    # Function for creating the tokenize
    def create_tokenizer(self, sentences):
        
        # Initialize the tokenizer
        tokenizer = tf.keras.preprocessing.text.Tokenizer()

        # Fit the tokenizer on the dataset
        tokenizer.fit_on_texts(sentences)

        return tokenizer
    
    # Function for fining the the sequence length
    def find_seq_length(self, dataset):

        # Sequence length
        seq_length = max(len(seq.split()) for seq in dataset)

        return seq_length
    
    # Function for finding the vocab size
    def find_vocab_size(self, tokenizer, dataset):

        # Fit the tokenizer on the dataset
        tokenizer.fit_on_texts(dataset)

        # Vocabulary size
        vocab_size = len(tokenizer.word_index) + 1

        return vocab_size 
    
    # Call function 
    def __call__(self, filename, **kwargs):

        # Load the clean dataset
        clean_dataset = pickle.load(open(filename, "rb"))

        # Reduce the dataset size
        dataset = clean_dataset[:self.n_sentences, :]

        # Include the "start" and "end" token to the sentences
        for i in range(dataset[:, 0].size):

            # Add the tokens
            dataset[i, 0] = "<START>" + dataset[i, 0] + "<EOS>"
            dataset[i, 1]  = "<START>" + dataset[i, 1] + "<EOS>"

        # Shuffle the dataset
        np.random.shuffle(dataset)

        # Split the dataset
        train = dataset[:int(self.n_sentences * self.train_split)]

        # Create the tokenizer for encoder
        encoder_tokenizer = self.create_tokenizer(train[:, 0])
        encoder_seq_length = self.find_seq_length(train[:, 0])
        encoder_vocab_size = self.find_vocab_size(encoder_tokenizer, train[:, 0])

        # Encode and pad the input sequences
        trainX = encoder_tokenizer.texts_to_sequence(train[:, 0])
        trainX = tf.keras.preprocessing.sequence.pad_sequences(trainX, maxlen=encoder_seq_length, padding="post")
        trainX = tf.convert_to_tensor(trainX)

        # Prepare the tokenizer for decoder
        decoder_tokenizer = self.create_tokenizer(train[:, 1])
        decoder_seq_length = self.find_seq_length(train[:, 1])
        decoder_vocab_size = self.find_vocab_size(decoder_tokenizer, train[:, 1])

        # Encode and pad the input sequences
        trainY = decoder_tokenizer.texts_to_sequences([train:, 1])
        trainY = tf.keras.preprocessing.sequence.pad_sequences(trainY, maxlen=decoder_seq_length, padding="post")
        trainY = tf.convert_to_tensor(trainY)

        return (trainX, trainY, train, encoder_seq_length, decoder_seq_length, encoder_vocab_size, decoder_vocab_size)
