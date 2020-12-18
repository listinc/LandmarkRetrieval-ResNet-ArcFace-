import os
import glob
import h5py

import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True
DATA_PATH = '/home/list-pc31/hdd/hdd_ext/2020_08_LANDMARK/LANDMARK_Dataset/'
data_path = '1차/서울시_split/'
img_path_1 = os.path.join(DATA_PATH, data_path+'train')
img_path_val = os.path.join(DATA_PATH, data_path+'val')

img_width = 224
img_height = 224

train_datagen = ImageDataGenerator(
    rescale=1./255
)
val_datagen = ImageDataGenerator(
    rescale=1./255
)

train_img_count = len(glob.glob(img_path_1 + "/*/*.JPG"))
val_img_count = len(glob.glob(img_path_val + "/*/*.JPG"))

train_g = train_datagen.flow_from_directory(
    img_path_1,
    target_size=(img_width, img_height),
    batch_size=train_img_count,
    shuffle=True,
    seed=123,
    color_mode='rgb',
    class_mode='categorical'
)
val_g = val_datagen.flow_from_directory(
    img_path_val,
    target_size=(img_width, img_height),
    batch_size=val_img_count,
    shuffle=True,
    seed=123,
    color_mode='rgb',
    class_mode='categorical'
)

train_x, train_y = train_g.next()
train_f = h5py.File(os.path.join(DATA_PATH, data_path+'train_save.hdf5'), 'w')
train_f['x'] = train_x
train_f['y'] = train_y
train_f.close()

val_x, val_y = val_g.next()
val_f = h5py.File(os.path.join(DATA_PATH, data_path+'val_save.hdf5'), 'w')
val_f['x'] = val_x
val_f['y'] = val_y
val_f.close()
