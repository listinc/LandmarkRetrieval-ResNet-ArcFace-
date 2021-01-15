import os
import gc
import sys

import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from PIL import ImageFile
from arcface_4 import dataset_train
from arcface_4.metrics import ArcFace
import tensorflow_datasets as tfds


ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_PATH = '/home/list-pc31/hdd/hdd_ext/2020_08_LANDMARK/LANDMARK_Dataset/'
data_path = '1차_2차_합/서울시_split/'
model_path = 'train_result/seoul_3'

batch_size = 16
epochs = 500
img_width = 224
img_height = 224

# 1차 27, 2차 282, 1차+2차 309
class_count = 309
train_count = 23484
val_count = 6180

train_dataset = dataset_train.load_tfrecord_dataset(
    tfrecord_name=os.path.join(DATA_PATH, data_path, 'train.tfrecord'),
    batch_size=batch_size, binary_img=True, class_n=class_count)

val_dataset = dataset_train.load_tfrecord_dataset(
    tfrecord_name=os.path.join(DATA_PATH, data_path, 'val.tfrecord'),
    batch_size=batch_size, binary_img=True, class_n=class_count)

steps_per_epoch = train_count // batch_size
val_steps = val_count // batch_size

input_shape = (img_width, img_height, 3)
resnet_model = ResNet101(
    input_shape=input_shape,
    weights='imagenet',
    include_top=False,
    pooling='avg',
    classifier_activation=None
)

weight_decay = 1e-4

input_tensor = resnet_model.input
y = Input(shape=(class_count,))
# y = Input([])

x = resnet_model.output
x = BatchNormalization()(x)
x = Dropout(.5)(x)
x = Flatten()(x)
x = Dense(512, activation=None, kernel_initializer='he_normal',
          kernel_regularizer=regularizers.l2(weight_decay))(x)
x = BatchNormalization()(x)
x = ArcFace(class_count, regularizer=regularizers.l2(weight_decay))(x, y)
# x = ArcFace(class_count, regularizer=regularizers.l2(weight_decay))(x)

model = Model(inputs=(input_tensor, y), outputs=x)
# model = Model(inputs=input_tensor, outputs=x)
# print(model.summary())
# sys.exit()
# 학습 중지 후 다시 학습할 때, 마지막으로 학습된 가중치 불러와서 사용
last_ckpt = tf.train.latest_checkpoint(model_path)
if not str(last_ckpt) == 'None':
    model.load_weights(last_ckpt)

model.compile(
    optimizer=tf.keras.optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True),
    loss='categorical_crossentropy',
    # loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

patience = 8
RR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience/2, min_lr=0.000001, verbose=1, mode='min')
ES = EarlyStopping(monitor='val_loss', patience=8, verbose=1)
MC = ModelCheckpoint(filepath=model_path+'/{epoch:02d}-{val_loss:.2f}.ckpt',
                     monitor='val_loss',
                     save_weights_only=True,
                     verbose=1)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    # [train_x, train_y],
    # validation_data=([val_x, val_y], val_y),
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    epochs=epochs,
    callbacks=[RR, ES, MC],
    use_multiprocessing=True,
    workers=8,
    batch_size=batch_size,
    verbose=1
)

tf.keras.models.save_model(model, model_path+'/model')

gc.collect()
