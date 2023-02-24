#%% Libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

import cv2
import tensorflow as tf
from tensorflow import keras
from keras import layers, losses, callbacks
from tensorflow_examples.models.pix2pix import pix2pix

from sklearn.model_selection import train_test_split
from IPython.display import clear_output

#%%
print(os.getcwd())

# %% 1.0 Load the data
# 1.1 Prepare an empty list for the images and mask
images = []
masks = []

# 1.2 Load the images using opencv
image_dir = os.path.join(
    os.getcwd(), "datasets", "data-science-bowl-2018-2", "train", "inputs"
)
for image_file in os.listdir(image_dir):
    img = cv2.imread(os.path.join(image_dir, image_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    images.append(img)

# 1.3 Load the masks
mask_dir = os.path.join(
    os.getcwd(), "datasets", "data-science-bowl-2018-2", "train", "masks"
)
for mask_file in os.listdir(mask_dir):
    mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (128, 128))
    masks.append(mask)

# %% 1.4 Convert the list into numpy array
images_np = np.array(images)
masks_np = np.array(masks)

# %% #1.5 Check some examples
plt.figure(figsize=(10, 10))
for i in range(1, 4):
    plt.subplot(1, 3, i)
    plt.imshow(images_np[i])
    plt.axis("off")
    plt.title("Image " + str(i))
plt.show()

plt.figure(figsize=(10, 10))
for i in range(1, 4):
    plt.subplot(1, 3, i)
    plt.imshow(masks_np[i])
    plt.axis("off")
    plt.title("Mask " + str(i))
plt.show()

# %% 2.0 Data preprocessing

# 2.1 Expand the mask dimension
masks_np_exp = np.expand_dims(masks_np, axis=-1)

# Check the mask output
print(np.unique(masks[0]))
# %% 2.2 Convert the mask values into class labels
converted_masks = np.round(masks_np_exp / 255).astype(np.int64)

# Check the mask output
print(np.unique(converted_masks[0]))

# %% 2.3 Normalize image pixels value
converted_images = images_np / 255.0
sample = converted_images[0]

# %% 2.4 Perform train-test split
SEED = 12345
x_train, x_test, y_train, y_test = train_test_split(
    converted_images, converted_masks, test_size=0.2, random_state=SEED
)

# %% 2.5 Convert the numpy arrays into tensor
"""
In this part of the code, the numpy arrays containing the images and 
masks are converted into TensorFlow datasets 
using the tf.data.Dataset.from_tensor_slices method. 
This method takes a numpy array and returns a tf.data.Dataset object.
"""
x_train_tensor = tf.data.Dataset.from_tensor_slices(x_train)
x_test_tensor = tf.data.Dataset.from_tensor_slices(x_test)
y_train_tensor = tf.data.Dataset.from_tensor_slices(y_train)
y_test_tensor = tf.data.Dataset.from_tensor_slices(y_test)

# %% 2.6 Combine the images and masks using zip
"""
The zip method is used to combine the images and masks datasets into a single dataset, 
where each element is a tuple containing an image and its corresponding mask.
"""
train_dataset = tf.data.Dataset.zip((x_train_tensor, y_train_tensor))
test_dataset = tf.data.Dataset.zip((x_test_tensor, y_test_tensor))

#%% [EXTRA] Create a subclass layer for data augmentation
"""
A custom layer called Augment is defined using the layers.Layer superclass. 
This layer applies random horizontal flipping to both the input images 
and their corresponding masks, using the RandomFlip layer from Keras. 
The call method is overridden to apply the augmentation to the inputs and labels.
"""


class Augment(layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        self.augment_inputs = layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = layers.RandomFlip(mode="horizontal", seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels


#%% 2.7 Convert into prefetch dataset
"""
The training dataset is then converted into a prefetch dataset using 
the cache, shuffle, batch, repeat, map, and prefetch methods.
"""
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
TRAIN_SIZE = len(train_dataset)
STEPS_PER_EPOCH = TRAIN_SIZE // BATCH_SIZE

"""
The cache method caches the dataset elements in memory to improve training performance.
The shuffle method shuffles the dataset elements randomly with a buffer size of 1000.
The batch method groups the elements into batches of size 16. The repeat method repeats the dataset infinitely.
The map method applies the Augment layer to each element of the dataset.
The prefetch method prefetches the next batch of elements asynchronously using tf.data.AUTOTUNE buffer size to optimize performance.
"""
train_batches = (
    train_dataset.cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# The test dataset is also batched into batches of 16 using the batch method.
test_batches = test_dataset.batch(BATCH_SIZE)

#%% 3.0 Visualize some examples
def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ["Input Image", "True Mask", "Predicted Mask"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    plt.show()


# %%
"""
for loop runs for 2 iterations using train_batches. 
For each iteration, it gets the first image and 
mask from the batch and displays them using the 
display function defined earlier. 
"""
for images, masks in train_batches.take(2):
    sample_image, sample_mask = images[0], masks[0]
    display([sample_image, sample_mask])

# %% 4.0 Create image segmentation model
"""
Use a pre-trained MobileNetV2 model as the feature extraction layers: 
The MobileNetV2 model is a pre-trained convolutional neural network 
that can be used as a feature extractor for images. 
The include_top=False argument means that 
the final fully connected layers of the model are excluded, 
so only the convolutional layers are used.
"""
base_model = keras.applications.MobileNetV2(
    input_shape=[128, 128, 3], include_top=False
)

# List down some activation layers
layer_names = [
    "block_1_expand_relu",  # 64x64
    "block_3_expand_relu",  # 32x32
    "block_6_expand_relu",  # 16x16
    "block_13_expand_relu",  # 8x8
    "block_16_project",  # 4x4
]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Define the feature extraction model
"""
The down_stack model is created using the Keras Functional API. 
The inputs to the model are the input image tensor, 
and the outputs are the output tensors from the activation layers listed in layer_names. 
The trainable=False argument means that the weights of the MobileNetV2 model layers 
are frozen and will not be updated during training.
"""
down_stack = keras.Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False

# Define the upsampling path
"""
The up_stack list contains a series of convolutional layers that are used 
to upsample the feature maps in the decoder path of the U-Net architecture.
"""
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),  # 32x32 -> 64x64
]

"""
 The unet_model function takes an output_channels argument, 
 which specifies the number of output channels (i.e., the number of classes) 
 for the segmentation mask. 
 
 The function starts with an input tensor of shape [128, 128, 3], 
 applies the feature extraction layers using the down_stack model, 
 and then starts the upsampling and skip connection process using the up_stack list. 
 
 The final layer is a Conv2DTranspose layer that produces an 
 output tensor of the same height and width as the input image, 
 but with output_channels number of channels.
"""


def unet_model(output_channels: int):
    inputs = layers.Input(shape=[128, 128, 3])
    # Apply functional API to construct U-Net
    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections(concatenation)
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model (output layer)
    last = layers.Conv2DTranspose(
        filters=output_channels,
        kernel_size=3,
        strides=2,
        padding="same",  # 64x64 -> 128x128
    )
    x = last(x)

    return keras.Model(inputs=inputs, outputs=x)


# %% Make use of the function to construct the entire U-Net
"""
OUTPUT_CLASSES is set to 2 because the goal is to perform binary segmentation of the images. 
This means that for each pixel in the input image, 
the model should output a prediction of either "object present" or "object not present". 
Therefore, there are only two possible output classes: 0 (background) and 1 (object present).
"""
OUTPUT_CLASSES = 2

"""
the unet_model() function is called to construct the entire U-Net architecture. 
The function takes an integer parameter output_channels which is the number of output classes
"""
model = unet_model(output_channels=OUTPUT_CLASSES)
# Compile the model
"""
The from_logits=True argument is passed to indicate that the model output is 
not normalized and requires a softmax activation.
"""
loss = losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
keras.utils.plot_model(model, show_shapes=True)

# %% Create functions to show predictions
"""
The create_mask function takes a predicted mask as input, 
applies tf.argmax to get the channel with the highest probability, 
adds an extra channel dimension, and returns the resulting mask.
"""


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


"""
The show_predictions function takes an optional dataset as input and a number of examples 
to display. If a dataset is given, it iterates over the first num examples, 
gets the predicted mask from the model, and displays the input image, true mask, 
and predicted mask using the display function defined earlier. 
If no dataset is given, it displays the same information for 
a single example sample_image and sample_mask that were defined earlier
"""


def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])

    else:
        display(
            [
                sample_image,
                sample_mask,
                create_mask(model.predict(sample_image[tf.newaxis, ...])),
            ]
        )


# %% Test out the show_prediction function
show_predictions()

# %% Create callback to help display results during model training
"""
DisplayCallback is defined to execute the on_epoch_end method at the end of each epoch. 
This method will clear the output console, 
then call the show_predictions function to display the predictions made by the model 
on a sample input image and its corresponding mask. 
It will also print out a message indicating the epoch number, for convenience
"""


class DisplayCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print("\nSample Prediction after epoch {}\n".format(epoch + 1))


#%% TensorBoard callbacks and model fitting

base_log_path = r"tensorboard_logs"
log_path = os.path.join(
    base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)
# Define the callbacks
display_callback = DisplayCallback()
tensorboard_callback = callbacks.TensorBoard(log_path)
# %% 5.0  Model Training
# Hyperparameters for model
EPOCHS = 10
VAL_SUBSPLITS = 5
VALIDATION_STEPS = len(test_dataset) // BATCH_SIZE // VAL_SUBSPLITS

history = model.fit(
    train_batches,
    validation_data=test_batches,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    callbacks=[display_callback, tensorboard_callback],
)

# %% 6.0 Model Deployment
show_predictions(test_batches, 3)
# %% Saving the model
model.save("saved_models/.model.h5")

#%% 7.0 Testing the model with test datasets
test_images = []
test_masks = []

# Load the test images using opencv
test_image_dir = os.path.join(
    os.getcwd(), "datasets", "data-science-bowl-2018-2", "test", "inputs"
)
for test_image_file in os.listdir(test_image_dir):
    test_img = cv2.imread(os.path.join(test_image_dir, test_image_file))
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    test_img = cv2.resize(test_img, (128, 128))
    test_images.append(test_img)

# Load the test masks
test_mask_dir = os.path.join(
    os.getcwd(), "datasets", "data-science-bowl-2018-2", "test", "masks"
)
for test_mask_file in os.listdir(test_mask_dir):
    test_mask = cv2.imread(
        os.path.join(test_mask_dir, test_mask_file), cv2.IMREAD_GRAYSCALE
    )
    test_mask = cv2.resize(test_mask, (128, 128))
    test_mask = np.expand_dims(test_mask, axis=-1)
    test_masks.append(test_mask)

# Convert the list of test images and masks into numpy arrays
test_images_np = np.array(test_images)
test_masks_np = np.array(test_masks)

# Normalize the pixel values of test images
test_images_np = test_images_np / 255.0

# Use the show_predictions method to visualize the predicted masks
show_predictions(
    tf.data.Dataset.from_tensor_slices((test_images_np, test_masks_np)).batch(1),
    num=10,
)
