from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import sys
import pickle
import io
import gc
import numpy as np
import tensorflow as tf

from Resnet.metrics import ArcFace, CosFace, SphereFace
from collections import defaultdict
from itertools import chain
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_PATH = '/home/list-pc31/hdd/hdd_ext/2020_08_LANDMARK/LANDMARK_Dataset/'
data_path = '1차_2차_합/서울시_split/'
train_path = 'train_result/seoul_1and2_test3/'

train_img_path = os.path.join(DATA_PATH, data_path+'train')

batch_size = 1
img_width = 224
img_height = 224

train_datagen = ImageDataGenerator(
    rescale=1./255
)
train_generator = train_datagen.flow_from_directory(
    train_img_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

model = tf.keras.models.load_model(train_path+'model')
model = Model(inputs=model.input[0], outputs=model.layers[-3].output)

features_train = model.predict(train_generator, verbose=1)
features_train /= np.linalg.norm(features_train, axis=1, keepdims=True)

val_img_path = os.path.join(DATA_PATH, data_path+'val')

val_datagen = ImageDataGenerator(
    rescale=1./255
)
val_generator = val_datagen.flow_from_directory(
    val_img_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

model = tf.keras.models.load_model(train_path+'/model')
model = Model(inputs=model.input[0], outputs=model.layers[-3].output)

features_val = model.predict(val_generator, verbose=1)
features_val /= np.linalg.norm(features_val, axis=1, keepdims=True)

data = {"id": list(train_generator.filenames)+list(val_generator.filenames),
        "features": np.concatenate((features_train, features_val), axis=0)}

f = open(train_path+'index.pickle', "wb")
f.write(pickle.dumps(data))
f.close()


test_img_path = os.path.join(DATA_PATH, data_path+'query')

test_datagen = ImageDataGenerator(
    rescale=1./255
)
test_generator = test_datagen.flow_from_directory(
    test_img_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

model = tf.keras.models.load_model(train_path+'/model')
model = Model(inputs=model.input[0], outputs=model.layers[-3].output)

features = model.predict(test_generator, verbose=1)
features /= np.linalg.norm(features, axis=1, keepdims=True)

data = {"id": list(test_generator.filenames), "features": features}

f = open(train_path+'test.pickle', "wb")
f.write(pickle.dumps(data))
f.close()

gc.collect()
