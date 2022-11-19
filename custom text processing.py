
# Import the libraries
import re, nltk, string
import tensorflow as tf
import numpy as np


# Class for preprocessing
class TextPreprocessingLayer(tf.keras.layers.Layer):

    #####################
    #    CONSTRUCTOR    #
    #####################

    # Constructor 
    def __init__(self, language="english", max_len=100, **kwargs):

        # Inherite parent's constructor
        super(TextPreprocessingLayer, self).__init__()

        # Initialize word2int and int2word
        self.word2int, self.int2word = self.initialize_word_dic()

        # Initialization
        self.language = language
        self.max_len = max_len


    #######################
    #    PREPROCESSING    #
    #######################

    # Function for preprocessing
    def preprocess(self, text):

        # Lowercase
        text = text.lower()

        # Remove the new line character
        text = text.replace("\n", "")
        
        # Remove unnecessary characters
        #text = re.sub(r"[/.,|?><;:±!@#$%^&*()_+=-]", " ", text)
        
        return text

    ############################
    #    WORD2INT, INT2WORD    #
    ############################

    # Function for initializing the word2int and int2word
    def initialize_word_dic(self):

        # int2word
        int2word = {idx: w.lower() for idx, w in enumerate(np.unique(nltk.corpus.words.words()))}
        int2word[max(int2word)+1] = "<SOS>"
        int2word[max(int2word)+1] = "<EOS>"
        int2word[max(int2word)+1] = "<UNK>"

        # word2int
        word2int = {w: idx for idx, w in int2word.items()}

        return word2int, int2word

    # Function for creating Word2Int and Int2Word
    def update_word_dic(self, text):

        # Convert list of strings into one string
        #text = " ".join(list_of_texts.ravel()).lower()                  

        # Tokenize
        tokens = nltk.tokenize.word_tokenize(text, language=self.language)   
        
        # Unique tokens
        unique_tokens = np.unique(tokens)                               

        # Loop over unique tokens
        for i_token in unique_tokens:

            # If token is not in word2int
            if i_token not in self.word2int:

                # Add to word2int, int2word
                self.word2int[i_token] = max(self.int2word)+1
                self.int2word[max(self.int2word)] = i_token


    #######################
    #    LABEL ENCODER    #
    #######################

    # Label encoding function; For converting words to numerical values
    def label_encoding(self, text):

        # Tokenize
        tokens = nltk.tokenize.word_tokenize(text, language=self.language)

        # Add staring/ending tag
        tokens.insert(0, "<SOS>")
        tokens.insert(len(tokens), "<EOS>")

        # Initialize a list
        encoded_tokens = []

        # Loop over tokens
        for i in tokens:

            # Append idx if it exist
            try: encoded_tokens.append(self.word2int[i])

            # Append <UNK> if it doesn't exist
            except: encoded_tokens.append(self.word2int["<UNK>"])

        return encoded_tokens

    #######################
    #    LABEL DECODER    #
    #######################

    # Label decoding function; For converting numerical values to words
    def label_decoding(self, tokens):

        # Initialize a list
        decoded_tokens = []

        # Loop over tokens
        for i in tokens:

            # Append word if it exist
            try: decoded_tokens.append(self.int2word[i])

            # Append <UNK> if it doesn't exist
            except: decoded_tokens.append("<UNK>")

        return decoded_tokens

    #################
    #    PADDING    #
    #################

    # Function for padding the sequence
    def padding(self, label_encoded_seq):

        # Pad the sequences
        padded_seq = tf.keras.preprocessing.sequence.pad_sequences(sequences=label_encoded_seq, maxlen=self.max_len, padding="post", truncating="post")

        return padded_seq

    ##########################
    #    ONE-HOT ENCODING    #
    ##########################

    # One-hot encoding; Function for converting numerical values to binary vectors
    def one_hot_encoding(self, label_encoded_seq):

        # Initialize a list for one hot encoded
        one_hot_encoded = []
        
        # Loop over each token in the sequence
        for i_token in label_encoded_seq:

            # Initialize zero vector
            #token = np.zeros(shape=length)
            token = [0 for _ in range(max(self.int2word)+1)]
            
            # Set one
            token[i_token] = 1

            # Append to the list
            one_hot_encoded.append(token)

        #return np.array(one_hot_encoded)
        return one_hot_encoded

    ##########################
    #    ONE-HOT DECODING    #
    ##########################

    # Function for one hot decoding
    def one_hot_decoding(self, tokens):

        # Initialize a list for one hot decoded
        one_hot_decoded = []
            
        # Loop over each token in the sequence
        for i_token in tokens:

            # Append to the list
            one_hot_decoded.append(np.argmax(i_token))

        #return np.array(one_hot_decoded)
        return one_hot_decoded
    
    # TODO: Function for decoding

    ##############
    #    CALL    #
    ##############

    # Function for call function
    def call(self, inputs, encoding_decoding):

        # TODO: Assert shape (shape of 2D as we getting batches)

        # If encoding
        if encoding_decoding=="encoding":
            
            # Preprocess the text
            inputs = self.preprocess(inputs)

            # Update word dictionary
            self.update_word_dic(inputs)

            # Label encoding
            label_encoded_seq = self.label_encoding(inputs)

            # Padding
            padded_seq = self.padding([label_encoded_seq])

            # One hot encoding
            one_hot_seq = self.one_hot_encoding(label_encoded_seq)

            return np.array(one_hot_seq)

        # If decoding
        elif encoding_decoding=="decoding":

            # One-hot decoding
            label_encoded = self.one_hot_decoding(inputs)

            # TODO: Convert digits into words

            return label_encoded


# # Initialize the text processing class
# processing_layer = TextPreprocessingLayer()

# # Process a sample text
# output = processing_layer("Hello, my name is John. I am 20 years old.", encoding_decoding="encoding")
# print(output.shape)
# print(output)



# Function for initial vocabualry
def initialize_word_vectors():

    # Initialize unique tokens
    # words_space_1 = list(english_words_set)
    words_space_2 = list(nltk.corpus.words.words())
    tags = ["[SEP]", "[CLS]", "[PAD]", "[MASK]", "[UNK]", "[EOS]", "[SOS]", "[START]", "[END]"]
    alphabets = list(string.ascii_lowercase)
    punctuation = list(string.punctuation)

    # List of all tokens
    tokens = words_space_2 + tags + alphabets + punctuation # + words_space_1

    # Lowercase all tokens
    tokens = [token.lower() for token in tokens]

    # Unique tokens
    unique_tokens = list(set(tokens))

    # Word2int
    #word2int = {i_word: i_idx for i_idx, i_word in enumerate(unique_tokens)}

    # Int2word
    #int2word = {i_idx: i_word for i_word, i_idx in word2int.items()}

    return unique_tokens