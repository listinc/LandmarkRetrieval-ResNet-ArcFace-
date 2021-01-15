import tensorflow as tf
import tensorflow_datasets as tfds


def _parse_tfrecord(class_n, binary_img=False, is_ccrop=False):
    def parse_tfrecord(tfrecord):
        if binary_img:
            features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                        'image/filename': tf.io.FixedLenFeature([], tf.string),
                        'image/encoded': tf.io.FixedLenFeature([], tf.string)}
            x = tf.io.parse_single_example(tfrecord, features)
            x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
        else:
            features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                        'image/img_path': tf.io.FixedLenFeature([], tf.string)}
            x = tf.io.parse_single_example(tfrecord, features)
            image_encoded = tf.io.read_file(x['image/img_path'])
            x_train = tf.image.decode_jpeg(image_encoded, channels=3)

        y_train = tf.cast(x['image/source_id'], tf.int32)

        x_train = _transform_images(is_ccrop=is_ccrop)(x_train)
        y_train = _transform_targets(y_train, class_n)

        return (x_train, y_train)
    return parse_tfrecord


def _transform_images(is_ccrop=False):
    def transform_images(x_train):
        x_train = tf.image.resize(x_train, (224, 224))
        x_train = x_train / 255
        return x_train
    return transform_images


def _transform_targets(y_train, class_n):
    y_train = tf.one_hot(y_train, class_n)
    return y_train


def _get_filename():
    def get_filename(tfrecord):
        features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                    'image/filename': tf.io.FixedLenFeature([], tf.string),
                    'image/encoded': tf.io.FixedLenFeature([], tf.string)}
        x = tf.io.parse_single_example(tfrecord, features)
        filename = tf.cast(x['image/filename'], tf.string)

        return filename
    return get_filename


def load_tfrecord_dataset(tfrecord_name, batch_size, class_n,
                          binary_img=False, shuffle=True, buffer_size=10240,
                          is_ccrop=False):
    """load dataset from tfrecord"""
    raw_dataset = tf.data.TFRecordDataset(tfrecord_name)
    # raw_dataset = raw_dataset.repeat()

    if shuffle:
        raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size)
    dataset = raw_dataset.map(
        _parse_tfrecord(class_n=class_n, binary_img=binary_img, is_ccrop=is_ccrop),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    filename = raw_dataset.map(_get_filename())

    return dataset, filename
