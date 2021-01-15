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

from arcface_4.dataset_test import load_tfrecord_dataset
from arcface_4.metrics import ArcFace

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_PATH = '/home/list-pc31/hdd/hdd_ext/2020_08_LANDMARK/LANDMARK_Dataset/'
data_path = ''
train_path = 'train_result/3/'
tfrecord_file = 'query.tfrecord'
feature_file = 'query_feature.pickle'


batch_size = 1
img_width = 224
img_height = 224

class_count = 4146

index_dataset, index_filename = load_tfrecord_dataset(
    tfrecord_name=os.path.join(DATA_PATH, data_path, tfrecord_file),
    batch_size=batch_size, binary_img=True, class_n=class_count, shuffle=False)

print('filename to list --- start')
index_filenames = []
for element in index_filename:
    e = element.numpy().decode('utf-8')
    index_filenames.append(e)

print('filename to list --- ok', len(index_filenames))

model = tf.keras.models.load_model(train_path+'model')
model = Model(inputs=model.input[0], outputs=model.layers[-3].output)

features = model.predict(
    index_dataset,
    verbose=1)
features /= np.linalg.norm(features, axis=1, keepdims=True)

data = {"id": index_filenames, "features": features}

f = open(train_path+feature_file, "wb")
f.write(pickle.dumps(data))
f.close()

gc.collect()
