# -*- coding: utf-8 -*-

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Conv2D
from keras.layers import BatchNormalization, GlobalAveragePooling2D
from keras.layers import MaxPooling2D, SeparableConv2D
from keras.layers import Input
from keras import layers
from keras.models import Model
from keras.regularizers import l2
import utils_body
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import backend as K
K.set_image_dim_ordering('tf')



# --- Data ---
images, labels = utils_body.load_FABO()

images = utils_body.preprocess_input(images)

# Divide data into train and validation subsets
images_train, images_val, labels_train, labels_val = train_test_split(
        images, labels, test_size = 0.25, shuffle = True)



# --- Parameters ---
# Batch size - the number of training examples in one forward/backward pass
batch_size = 32
# One epoch - one forward pass and one backward pass of all the training examples
num_epochs = 200 
input_shape = (96, 128, 1)
# Option for progress update
verbose = 1
num_classes = 7
# Number of epochs with no improvement after which training will be stopped
patience = 20
base_path = 'Model/'
l2_regularization = 0.01


# Artificially increases the dataset by performing variations of images in the 
# dataset by using horizontal/vertical flips, rotations, variations in 
# brightness of images, horizontal/vertical shifts etc.
data_generator = ImageDataGenerator(featurewise_center = False,
                                    featurewise_std_normalization = False,
                                    rotation_range = 10,
                                    width_shift_range = 0.1,
                                    height_shift_range = 0.1,
                                    zoom_range = 0.1,
                                    horizontal_flip = True)

# Regularizers allow to apply penalties on layer parameters or layer activity 
# during optimization. These penalties are incorporated in the loss function 
# that the network optimizes.
regularization = l2(l2_regularization)


# --- Base ---
# Input() is used to instantiate a Keras tensor. A tensor allows us to build 
# a Keras model.
img_input = Input(input_shape)
# Conv2D - This layer creates a convolution kernel that is convolved with the 
# layer input to produce a tensor of outputs.
# filters: Integer, the dimensionality of the output space (i.e. the number of 
# output filters in the convolution).
# kernel_size: An integer or tuple/list of 2 integers, specifying the height 
# and width of the 2D convolution window. 
# strides: An integer or tuple/list of 2 integers, specifying the strides of 
# the convolution along the height and width. Can be a single integer to specify 
# the same value for all spatial dimensions.
# padding: valid - without padding
x = Conv2D(8, (3, 3), strides = (1, 1), kernel_regularizer = regularization,
           use_bias = False)(img_input)
# Normalize the activations of the previous layer at each batch, i.e. applies a 
# transformation that maintains the mean activation close to 0 and the activation 
# standard deviation close to 1.
x = BatchNormalization()(x)
# Applies an activation function to an output. 
# activation: name of activation function to use. Rectified Linear Unit.
x = Activation('relu')(x)

x = Conv2D(8, (3, 3), strides = (1, 1), kernel_regularizer = regularization,
           use_bias = False)(img_input)
x = BatchNormalization()(x)
x = Activation('relu')(x)



# --- Module 1 ---
residual = Conv2D(16, (1,1), strides = (2, 2), padding = 'same', 
                  use_bias = False)(x)
residual = BatchNormalization()(residual)

# Separable convolutions consist in first performing a depthwise spatial  
# convolution (which acts on each input channel separately) followed by a  
# pointwise convolution which mixes together the resulting output channels. 
# Separable convolutions can be understood as a way to factorize a convolution
# kernel into two smaller kernels, or as an extreme version of an Inception block.
x = SeparableConv2D(16, (3, 3), padding = 'same', 
                    kernel_regularizer = regularization, use_bias = False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = SeparableConv2D(16, (3, 3), padding = 'same', 
                    kernel_regularizer = regularization, use_bias = False)(x)
x = BatchNormalization()(x)

# Max pooling operation for spatial data.
# pool_size: tuple, factors by which to downscale (vertical, horizontal).  
# (2, 2) will halve the input in both spatial dimension.
x = MaxPooling2D((3, 3), strides = (2, 2), padding = 'same')(x)

# Returns a tensor, the sum of the inputs.
x = layers.add([x, residual])



# --- Module 2 ---
residual = Conv2D(32, (1, 1), strides = (2, 2), padding = 'same', 
                  use_bias = False)(x)
residual = BatchNormalization()(residual)

x = SeparableConv2D(32, (3, 3), padding = 'same',
                    kernel_regularizer = regularization, use_bias = False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = SeparableConv2D(32, (3, 3), padding = 'same',
                    kernel_regularizer = regularization, use_bias = False)(x)
x = BatchNormalization()(x)

x = MaxPooling2D((3, 3), strides = (2, 2), padding = 'same')(x)
x = layers.add([x, residual])



# --- Module 3 ---
residual = Conv2D(64, (1, 1), strides = (2, 2), padding = 'same',
                  use_bias = False)(x)
residual = BatchNormalization()(residual)

x = SeparableConv2D(64, (3, 3), padding = 'same',
                    kernel_regularizer = regularization, use_bias = False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = SeparableConv2D(64, (3, 3), padding = 'same',
                    kernel_regularizer = regularization, use_bias = False)(x)
x = BatchNormalization()(x)

x = MaxPooling2D((3, 3), strides = (2, 2), padding = 'same')(x)
x = layers.add([x, residual])



# --- Module 4 ---
residual = Conv2D(128, (1, 1), strides = (2, 2), padding='same', 
                  use_bias = False)(x)
residual = BatchNormalization()(residual)

x = SeparableConv2D(128, (3, 3), padding = 'same',
                    kernel_regularizer = regularization, use_bias = False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = SeparableConv2D(128, (3, 3), padding = 'same',
                    kernel_regularizer = regularization, use_bias = False)(x)
x = BatchNormalization()(x)

x = MaxPooling2D((3, 3), strides = (2, 2), padding = 'same')(x)
x = layers.add([x, residual])



x = Conv2D(num_classes, (3, 3), padding = 'same')(x)
# Global average pooling operation for spatial data.
x = GlobalAveragePooling2D()(x)
output = Activation('softmax', name = 'predictions')(x)

model = Model(img_input, output)
# Configures the model for training.
# optimizer: String (name of optimizer) or optimizer instance. Adam optimizer.
# loss: String (name of objective function) or objective function. 
# metrics: List of metrics to be evaluated by the model during training and testing.
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])
# Prints a summary representation of the model.
model.summary()



# ---Callbacks---
# A callback is a set of functions to be applied at given stages of the training 
# procedure. You can use callbacks to get a view on internal states and statistics 
# of the model during training. 
log_file_path = base_path + 'emotion_training.log'
# Callback that streams epoch results to a csv file.
csv_logger = CSVLogger(log_file_path, append = False)
# Stop training when a monitored quantity has stopped improving.
early_stop = EarlyStopping('val_loss', patience = patience)
# Reduce learning rate when a metric has stopped improving.
reduce_lr = ReduceLROnPlateau('val_loss', factor = 0.1, patience = int(patience/4), 
                              verbose = 1)
trained_models_path = base_path + 'mini_XCEPTION'
model_names = trained_models_path + '.{epoch:02d}-{val_acc:0.2f}.hdf5'
# Save the model after every epoch.
# save_best_only: if save_best_only=True, the latest best model according to 
# the quantity monitored will not be overwritten.
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose = 1, 
                                   save_best_only = True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]


# fit_generator - Trains the model on data generated batch-by-batch by a Python 
# generator
# flow - Takes data & label arrays, generates batches of augmented data.
hist = model.fit_generator(data_generator.flow(images_train, labels_train, batch_size),
                           steps_per_epoch = len(images_train) / batch_size, 
                           epochs = num_epochs, verbose = 1, callbacks = callbacks,
                           validation_data = (images_val, labels_val))





epochs = len(hist.history['loss'])

# Visualize loss and accuracy

train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['acc']
val_acc = hist.history['val_acc']
xc = range(epochs)

plt.figure(1, figsize = (7, 5))
plt.plot(xc, train_loss)
plt.plot(xc, val_loss)
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.title('Train loss vs validation loss')
plt.grid(True)
plt.legend(['train', 'validation'])
plt.style.use(['classic'])

plt.figure(2, figsize = (7, 5))
plt.plot(xc, train_acc)
plt.plot(xc, val_acc)
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.title('Train accuracy vs validation accuracy ')
plt.grid(True)
plt.legend(['train', 'validation'], loc = 4)
plt.style.use(['classic'])





