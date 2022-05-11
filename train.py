
# %%

# Imports

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications.vgg16 import VGG16 as PretrainedModel, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback

from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import os

# %%

# Init callback functions

checkpoint_path = "./checkpoints/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


class CustomCallback(Callback):
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start testing; got log keys: {}".format(keys))

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop testing; got log keys: {}".format(keys))

    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start predicting; got log keys: {}".format(keys))

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop predicting; got log keys: {}".format(keys))

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print(logs)
        print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))

# %%

# Function definitions


def get_class_names(folders):
    class_names = []
    for i in range(0, len(folders)):
        class_names.append(folders[i].split('\\')[-1])
    return class_names

# %%

# Init variable


train_path = './data_set/training'
valid_path = './data_set/validation'

IMAGE_SIZE = [200, 200]

image_files = glob(train_path + '/*/*.jpg')
valid_image_files = glob(valid_path + '/*/*.jpg')

folders = glob(train_path + '/*')

class_name_map = map(get_class_names, folders)
class_names = list(class_name_map)


# %%

# Load pre trained model

ptm = PretrainedModel(input_shape=IMAGE_SIZE +
                      [3], weights="imagenet", include_top=False)

ptm.trainable = False
K = len(folders)
x = Flatten()(ptm.output)
x = Dense(K, activation='softmax')(x)

# %%

# Create model

model = Model(inputs=ptm.input, outputs=x)

model.summary()

# %%

# Create generators

gen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
                         shear_range=0.1, zoom_range=0.2, horizontal_flip=True, preprocessing_function=preprocess_input)

batch_size = 128

train_generator = gen.flow_from_directory(
    train_path, shuffle=True, target_size=IMAGE_SIZE, batch_size=batch_size)

valid_generator = gen.flow_from_directory(
    valid_path, target_size=IMAGE_SIZE, batch_size=batch_size)

# %%

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# %%

# Train model

r = model.fit_generator(train_generator, validation_data=valid_generator, callbacks=[CustomCallback(), cp_callback], epochs=10, steps_per_epoch=int(
    np.ceil(len(image_files)/batch_size)), validation_steps=int(np.ceil(len(valid_image_files) / batch_size)))


# %%

print("Training complete.")

# %%

model.save('model/my_model')

print("Model saved")
