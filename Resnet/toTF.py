from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tqdm
import glob
import random
import tensorflow as tf

import pickle


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_example(img_str, source_id, filename):
    # Create a dictionary with features that may be relevant.
    feature = {'image/source_id': _int64_feature(source_id),
               'image/filename': _bytes_feature(filename),
               'image/encoded': _bytes_feature(img_str)}

    return tf.train.Example(features=tf.train.Features(feature=feature))


def main(_):
    dataset_path = "/home/list-pc31/hdd/hdd_ext/2020_08_LANDMARK/LANDMARK_Dataset/1차_2차_합/서울시_split/query/"
    output_path = "/home/list-pc31/hdd/hdd_ext/2020_08_LANDMARK/LANDMARK_Dataset/1차_2차_합/서울시_split/query.tfrecord"
    label_path = "/home/list-pc31/hdd/hdd_ext/2020_08_LANDMARK/LANDMARK_Dataset/1차_2차_합/서울시_split/query.pickle"

    if not os.path.isdir(dataset_path):
        logging.info('Please define valid dataset path.')
    else:
        logging.info('Loading {}'.format(dataset_path))

    labels = {}
    samples = []
    logging.info('Reading data list...')
    i = 0
    for id_name in tqdm.tqdm(os.listdir(dataset_path)):
        img_paths = glob.glob(os.path.join(dataset_path, id_name, '*.JPG'))
        for img_path in img_paths:
            filename = os.path.join(id_name, os.path.basename(img_path))
            samples.append((img_path, i, filename))
        labels[id_name] = i
        i += 1
    random.shuffle(samples)

    with open(label_path, 'wb') as fw:
        pickle.dump(labels, fw)

    logging.info('Writing tfrecord file...')
    with tf.io.TFRecordWriter(output_path) as writer:
        for img_path, id_name, filename in tqdm.tqdm(samples):
            tf_example = make_example(img_str=open(img_path, 'rb').read(),
                                      source_id=int(id_name),
                                      filename=str.encode(filename))
            writer.write(tf_example.SerializeToString())


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
