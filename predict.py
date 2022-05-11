
# %%

# Imports

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications.vgg16 import VGG16 as PretrainedModel
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import re

# %%

# Function definitions


def show_class_images(prediction_results, class_names):
    for i in range(0, len(class_names)):
        print("Showing results for ", class_names[i])
        class_data = [
            a for a in prediction_results if a['classification'] == class_names[i]]

        for j in range(0, len(class_data)):
            plt.imshow(image.load_img(class_data[j]['path']))
            plt.show()
        print("End ", class_names[i])


def plot_prediction_results(prediction_results, class_names):

    class_name_counts = []
    for i in range(0, len(class_names)):
        count = 0
        for j in range(0, len(prediction_results)):
            if prediction_results[j]['classification'] == class_names[i]:
                count += 1
        class_name_counts.append(count)

    y_pos = np.arange(len(class_names))
    performance = class_name_counts

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, class_names)
    plt.ylabel('Count')
    plt.title('Prediction results')

    plt.show()


def predict_image_files(predict_path, class_names, model):
    predict_image_files = glob(predict_path + '/*.jpg')
    predict_image_data = []
    for i in range(0, len(predict_image_files)):
        image_data = {
            'path': predict_image_files[i], 'predict_data': [], 'classification': ''}

        img = tf.keras.utils.load_img(
            predict_image_files[i], target_size=(IMAGE_SIZE[0], IMAGE_SIZE[1])
        )

        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        image_prediction = model.predict(
            img_array,
            batch_size=1,
            verbose=1,
            steps=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False
        )

        image_data['predict_data'] = image_prediction

        highest_prediction_value = -1
        classification = ''

        for j in range(0, len(class_names)):
            if image_prediction[0][j] > highest_prediction_value:
                highest_prediction_value = image_prediction[0][j]
                classification = class_names[j]

        image_data['classification'] = classification
        predict_image_data.append(image_data)

    return predict_image_data


def get_class_names(folders, train_path):
    output_class_names = []
    for i in range(0, len(folders)):
        output_class_names.append(re.sub('[^a-zA-Z0-9 \n\.]', '',
                                         folders[i].split(train_path)[-1]))
    return output_class_names

# Init variables


# %%

# only used to get class names
train_path = './data_set/training'

# actual prediction path
predict_path = './data_set/prediction'

folders = glob(train_path + '/*')

class_names = get_class_names(folders, train_path)

IMAGE_SIZE = [200, 200]

# %%

folders[0].split('\\')[-1]

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

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# load from my desktop computer
checkpoint_path = "./checkpoints/cp.ckpt"
model.load_weights(checkpoint_path)

# %%

prediction_results = predict_image_files(
    predict_path=predict_path, model=model, class_names=class_names)

# %%

plot_prediction_results(
    prediction_results=prediction_results, class_names=class_names)


# %%

show_class_images(prediction_results=prediction_results,
                  class_names=class_names)

# %%
