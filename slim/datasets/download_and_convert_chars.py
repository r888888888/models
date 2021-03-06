import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from pprint import pprint
import re
import os
import urllib.request
import tensorflow as tf
import math
import random
import sys
from datasets import dataset_utils
from itertools import islice
from pathlib import Path

_VALIDATION_PERCENTAGE = 0.9
_RANDOM_SEED = 0
_NUM_SHARDS = 5

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.                    
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(
      self._decode_jpeg,
      feed_dict={self._decode_jpeg_data: image_data}
    )
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'chars_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)

def _tag_tokenizer(x):
  return x.split()

def _convert_dataset(split_name, hashes, class_names_to_ids, dataset_dir):
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(hashes) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(hashes))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(hashes), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(_image_path(dataset_dir, hashes[i]), 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            class_name = _read_label(dataset_dir, hashes[i])
            class_id = class_names_to_ids[class_name]

            example = dataset_utils.image_to_tfexample(
                image_data, b'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()

def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True

def _delete_old_images(dataset_dir, hashes):
  for file in Path(os.path.normpath(os.path.join(dataset_dir, "..", "images"))).iterdir():
    if file.is_file() and str(file) not in hashes:
      print("deleting", str(file))
      file.unlink()

def _download_images(dataset_dir):
  data = pd.read_csv(os.path.join(dataset_dir, "posts_chars.csv"))
  cv = CountVectorizer(min_df=0.002, tokenizer=_tag_tokenizer)
  cv.fit(data["character"])
  chars = set(cv.vocabulary_.keys())
  hashes = set()

  with open(os.path.join(dataset_dir, "num_classes.txt"), "w") as f:
    f.write(str(len(chars)))

  for index, row in data.iterrows():
    md5 = row["md5"]
    url = row["url"]
    char = row["character"]
    if re.search(r".jpg", url) and char in chars:
      local_path = _image_path(dataset_dir, md5)
      label_path = _label_path(dataset_dir, md5)
      hashes.add(md5)
      if not os.path.isfile(local_path):
        print("downloading", url)
        urllib.request.urlretrieve(url, local_path)
      with open(label_path, "w") as f:
        f.write(char)

  _delete_old_images(dataset_dir, hashes)
  return (list(hashes), chars)

def _label_path(dataset_dir, hash):
  return os.path.normpath(os.path.join(dataset_dir, "..", "image_labels/{}.txt".format(hash)))

def _read_label(dataset_dir, hash):
  with open(_label_path(dataset_dir, hash), "r") as f:
    return f.read()

def _image_path(dataset_dir, hash):
  return os.path.normpath(os.path.join(dataset_dir, "..", "images/{}.jpg".format(hash)))

def run(dataset_dir):
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  if _dataset_exists(dataset_dir):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  hashes, class_names = _download_images(dataset_dir)
  class_names_to_ids = dict(zip(class_names, range(len(class_names))))

  # Divide into train and test:
  random.seed(_RANDOM_SEED)
  random.shuffle(hashes)
  partition = int(len(hashes) * _VALIDATION_PERCENTAGE)
  training_hashes = hashes[partition:]
  validation_hashes = hashes[:partition]

  # First, convert the training and validation sets.
  _convert_dataset('train', training_hashes, class_names_to_ids,
                   dataset_dir)
  _convert_dataset('validation', validation_hashes, class_names_to_ids,
                   dataset_dir)

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  print('\nFinished converting the chars dataset!')