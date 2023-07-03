
# Turn of the tensorlfow logging
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import the libraries
import pickle, time
import numpy as np
import tensorflow as tf
from transformer import TransformerModel
from dataset import PrepareDataset
from hyperparameters import *

# Function for calculating the loss
def loss_function(target, prediction):
    """
    This function calculates the loss between the target and the prediction.

    PARAMETERS
    ==========================
        - target (tf.Tensor): the target tensor
        - prediction (tf.Tensor): the prediction tensor

    RETURNS
    ==========================
        - loss (tf.Tensor): the loss between the target and the prediction
    """

    # Mask the padding values
    mask = tf.math.logical_not(tf.math.equal(target, 0))
    mask = tf.cast(mask, tf.float32)

    # Computer the sparse categorical cross entropy loss on the unmasked values
    loss = tf.keras.losses.sparse_categorical_crossentropy(target, prediction, from_logits=True) * mask

    # Calculate the mean loss over the unmasked values
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)

    return loss


# Function for calculating the accuracy
def accuracy_function(target, prediction):
    """
    Function for calculating the accuracy between the target and the prediction.

    PARAMETERS
    ==========================
        - target (tf.Tensor): the target tensor
        - prediction (tf.Tensor): the prediction tensor

    RETURNS
    ==========================
        - out (tf.Tensor): the accuracy between the target and the prediction
    """

    # Mask the padding values
    mask = tf.math.logical_not(tf.math.equal(target, 0))
    
    # Calculate accuracy and apply the padding mask
    accuracy = tf.equal(target, tf.argmax(prediction, axis=2))
    accuracy = tf.math.logical_and(mask, accuracy)

    # Cast the accuracy from boolean to float32
    mask = tf.cast(mask, dtype=tf.float32)
    accuracy = tf.cast(accuracy, dtype=tf.float32)

    # Calculate the mean accuracy over the unmasked values
    out = tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    return out


# Class for scheduling the learning ear
class LearningRateScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    This class schedules the learning rate.

    PARAMETERS
    ==========================
        - d_model (int): the model's dimensionality
        - warmup_steps (int): the number of warmup steps
        
    RETURNS
    ==========================
        - learning_rate (tf.Tensor): the learning rate
    """
    
    # Constructor function
    def __init__(self, d_model, warmup_steps=4_000, **kwargs):

        # Inherite the parent's constructor
        super().__init__(**kwargs)

        # Initializations
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    # Call function
    def __call__(self, step_num):

        # Cast step_num to float
        step_num = tf.cast(step_num, tf.float32)

        # Linearly increase the learning rate for the first warmup_steps times, then decrease it
        arg1 = step_num ** -0.5
        arg2 = step_num * (self.warmup_steps ** -1.5)

        # Learning rate
        learning_rate = (self.d_model ** -0.5) * tf.math.minimum(arg1, arg2)

        return learning_rate
    
    
# Function for the training step (to spped up the training process)
@tf.function
def train_step(encoder_input, decoder_input, decoder_output):
    """
    This function performs a training step.

    PARAMETERS
    ==========================
        - encoder_input (tf.Tensor): the encoder input
        - decoder_input (tf.Tensor): the decoder input
        - decoder_output (tf.Tensor): the decoder output

    RETURNS
    ==========================
        - None
    """

    # Initialize the gradient tape
    with tf.GradientTape() as tape:

        # Forward pass (to make predictions)
        prediction = model(encoder_input, decoder_input, training=True)

        # Calculate loss
        loss = loss_function(decoder_output, prediction)

        # Calculate accuracy
        accuracy = accuracy_function(decoder_output, prediction)

    # Fetch the gradients of the trainable variables with respect to the training loss
    gradients = tape.gradient(loss, model.trainable_weights)

    # Apply the gradients to the optimizer so it can update the model accordingly
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    # Update the metrics
    train_loss(loss)
    train_accuracy(accuracy)


# Instantiate the optimizer with the learning rate scheduler
optimizer = tf.keras.optimizers.Adam(LearningRateScheduler(d_model), beta_1, beta_2, epsilon)

# Prepare the dataset
dataset = PrepareDataset()
train_x, train_y, val_x, val_y, train, val, encoder_sequence_length, decoder_sequence_length, encoder_vocabulary_size, decoder_vocabulary_size = dataset("./implementations from scratch/dataset/english-german-both.pkl")

# Convert to tf.data
train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_dataset = train_dataset.batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
val_dataset = val_dataset.batch(batch_size)

# Instantiate the model
model = TransformerModel(encoder_vocabulary_size, decoder_vocabulary_size, encoder_sequence_length, decoder_sequence_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)

# Include the metrics monitoring
train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")
val_loss = tf.keras.metrics.Mean(name="val_loss")

# Checkpoint object and manager (for managing multiple checkpoints)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, "./checkpoints", max_to_keep=None)

# Initialize lists for stroing the losses
train_loss_d = {}
val_loss_d = {}

# Loop over the epochs
for i_epoch in range(epochs):

    # Reset the metrics
    train_loss.reset_states()
    train_accuracy.reset_states()
    val_loss.reset_states()

    # Report 
    print(f"Epoch {i_epoch + 1}/{epochs}" + "\n===========================================")

    # Start a timer
    start_time = time.time()

    # Loop over the training batches
    for i_step, (train_batch_x, train_batch_y) in enumerate(train_dataset):

        # Define the encoder/decoder input/output
        encoder_input = train_batch_x[:, 1:]
        decoder_input = train_batch_y[:, :-1]
        decoder_output = train_batch_y[:, 1:]

        # Perform one training step
        train_step(encoder_input, decoder_input, decoder_output)

        # Report
        if (i_step % 50 == 0):
            print(f"Step {i_step}, Loss = {train_loss.result():.4f}, Accuracy = {train_accuracy.result():.4f}")

    # Loop over the validation batches
    for i_step, (val_batch_x, val_batch_y) in enumerate(val_dataset):

        # Define the encoder/decoder input/output
        encoder_input = val_batch_x[:, 1:]
        decoder_input = val_batch_y[:, :-1]
        decoder_output = val_batch_y[:, 1:]

        # Forward pass (to make predictions)
        prediction = model(encoder_input, decoder_input, training=False)

        # Calculate the loass
        loss = loss_function(decoder_output, prediction)

        # Update the metrics
        val_loss(loss)

        # Reoirt
        if (i_step % 50 == 0):
            print(f"Step {i_step}, Validation Loss = {val_loss.result():.4f}")


    # Report
    print(f"Training Loss = {train_loss.result():.4f}, Training Accuracy = {train_accuracy.result():.4f}, Validation Loss = {val_loss.result():.4f}")

    # Save the checkpoint after each epoch
    if (i_epoch+1) % 1 == 0:

        # Save the checkpoint
        save_path = checkpoint_manager.save()

        # Report
        print(f"Checkpoint saved at {save_path}.")

        # Save the weights
        model.save_weights("weights/wghts" + str(i_epoch + 1) + ".ckpt")

        # Report
        train_loss_d[i_epoch] = train_loss.result()
        val_loss_d[i_epoch] = val_loss.result()

# Save the loss values
with open("./train_loss.pkl", "wb") as file:  pickle.dump(train_loss_d, file)
with open("./val_loss.pkl", "wb") as file:  pickle.dump(val_loss_d, file)



