import glob
from io import BytesIO
import json
import os
import re

import boto3
import numpy as np
import tensorflow as tf


def get_train_val_iterator_from_npy(folder_path, batch_size, pre_fix="/train_*"):
    filenames = glob.glob(folder_path + pre_fix)
    np_list = []
    for npy_file in filenames:
        np_arrays = np.load(npy_file).astype(np.float32)
        np_list.append(np_arrays)
    features = np.concatenate(np_list, axis=0)
    print("features.shape:", features.shape)
    # batchs_per_epoch = features.shape[0]/batch_size
    full_dataset = tf.data.Dataset.from_tensor_slices(features)
    full_dataset = full_dataset.shuffle(10 * batch_size + 1000)
    DATASET_SIZE = features.shape[0]
    train_size = int(0.85 * DATASET_SIZE)
    val_size = int(0.15 * DATASET_SIZE)
    print("train_size: ", train_size, "val_size: ", val_size)
    batchs_per_train = int(train_size / batch_size) + 1
    batchs_per_val = int(val_size / batch_size) + 1
    train_dataset = full_dataset.take(train_size)
    val_dataset = full_dataset.skip(val_size)

    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(1)
    train_iterator = train_dataset.make_one_shot_iterator()

    val_dataset = val_dataset.cache()
    val_dataset = val_dataset.repeat()
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.prefetch(1)
    val_iterator = val_dataset.make_one_shot_iterator()

    return [train_iterator, val_iterator, batchs_per_train, batchs_per_val]


def get_iterator_from_npy(folder_path, batch_size, pre_fix="/train_*"):
    filenames = glob.glob(folder_path + pre_fix)
    np_list = []
    for npy_file in filenames:
        np_arrays = np.load(npy_file).astype(np.float32)
        np_list.append(np_arrays)
    features = np.concatenate(np_list, axis=0)
    print(features.shape)
    batchs_per_epoch = features.shape[0] / batch_size
    dataset = tf.data.Dataset.from_tensor_slices((features))
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    iterator = dataset.make_one_shot_iterator()
    return iterator, batchs_per_epoch


def _count_record(files):
    c = 0
    for fn in files:
        for record in tf.python_io.tf_record_iterator(fn):
            c += 1
    return c


def decode(serialized_example, decode_key, n_input=None):
    features = tf.parse_single_example(
        serialized_example,
        features={decode_key: tf.FixedLenFeature([n_input], dtype=tf.float32)})
    return features[decode_key]


def get_iterator_from_tfrecords(file_path, batch_size, decode_key, n_input, pre_fix, is_train):
    filenames = glob.glob(os.path.join(file_path, pre_fix))
    train_samples = _count_record(filenames)
    batchs_per_epoch = int(train_samples / batch_size) + 1
    # dataset = tf.data.TFRecordDataset(filenames)
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=32)
    dataset = dataset.map(
        lambda x: decode(x, decode_key=decode_key, n_input=n_input), num_parallel_calls=32)

    if is_train:
        dataset = dataset.shuffle(1000 + 3 * batch_size)
        dataset = dataset.cache()
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(4)
    iterator = dataset.make_one_shot_iterator()
    return iterator, batchs_per_epoch


def get_train_val_iterator_from_tfr_in_local(folder_path, batch_size, decode_key, n_input, n_row, pre_fix, cache=False):
    filenames = glob.glob(os.path.join(folder_path, pre_fix))
    # DATASET_SIZE = _count_record(filenames)
    DATASET_SIZE = n_row
    full_dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=32)
    full_dataset = full_dataset.shuffle(100 * batch_size + 1000)
    full_dataset = full_dataset.map(
        lambda x: decode(x, decode_key=decode_key, n_input=n_input), num_parallel_calls=32)
    train_size = int(0.99 * DATASET_SIZE)
    val_size = int(0.01 * DATASET_SIZE)
    print("train_size: ", train_size, "val_size: ", val_size)
    batchs_per_train = int(train_size / batch_size) + 1
    batchs_per_val = int(val_size / batch_size) + 1
    train_dataset = full_dataset.take(train_size)
    val_dataset = full_dataset.skip(train_size)

    if cache:
        train_dataset = train_dataset.cache()
        val_dataset = val_dataset.cache()

    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(4)
    train_iterator = train_dataset.make_one_shot_iterator()

    val_dataset = val_dataset.repeat()
    val_dataset = val_dataset.batch(10 * batch_size)
    val_dataset = val_dataset.prefetch(4)
    val_iterator = val_dataset.make_one_shot_iterator()

    return [train_iterator, val_iterator, batchs_per_train, batchs_per_val]


def get_train_val_iterator_from_tfr_in_s3(
        folder_path, batch_size, decode_key, n_input, n_rows, pre_fix, cache=False):
    print("folder_path:", folder_path)
    result = re.search("s3:\/\/(.+?)\/(.*)", folder_path)
    bucket_name = result.group(1)
    tfr_path = result.group(2)

    client = boto3.client('s3')
    response = client.list_objects(Bucket=bucket_name,
                                   Prefix=os.path.join(tfr_path, pre_fix))
    # print "response:", response
    filenames = [os.path.join('s3://', bucket_name, item['Key']) for item in response['Contents']]
    print("filenames length:", len(filenames))
    print(filenames[:2])
    # filenames = glob.glob(os.path.join(folder_path,pre_fix))
    DATASET_SIZE = round(n_rows / 100)
    full_dataset = tf.data.TFRecordDataset(filenames[:2], num_parallel_reads=8)
    full_dataset = full_dataset.shuffle(100 * batch_size + 1000)
    full_dataset = full_dataset.map(
        lambda x: decode(x, decode_key=decode_key, n_input=n_input), num_parallel_calls=32)
    train_size = int(0.99 * DATASET_SIZE)
    val_size = int(0.01 * DATASET_SIZE)
    print("train_size: ", train_size, "val_size: ", val_size)
    batchs_per_train = int(train_size / batch_size) + 1
    batchs_per_val = int(val_size / batch_size) + 1
    train_dataset = full_dataset.take(train_size)
    val_dataset = full_dataset.skip(train_size)

    if cache:
        train_dataset = train_dataset.cache()
        val_dataset = val_dataset.cache()

    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(4)
    train_iterator = train_dataset.make_one_shot_iterator()

    val_dataset = val_dataset.repeat()
    val_dataset = val_dataset.batch(10 * batch_size)
    val_dataset = val_dataset.prefetch(4)
    val_iterator = val_dataset.make_one_shot_iterator()

    return [train_iterator, val_iterator, batchs_per_train, batchs_per_val]


def get_train_val_iterator_from_tfr(folder_path, batch_size, decode_key,
                                    n_input, n_rows, pre_fix, cache=False):
    if "s3://" in folder_path:
        train_iter, val_iter, batchs_train, batchs_val = \
            input_utils.get_train_val_iterator_from_tfr_in_s3(
                input_path, batch_size, tfr_feature_key, n_feature,
                n_row, pre_fix, data_cache)
    else:
        train_iter, val_iter, batchs_train, batchs_val = \
            input_utils.get_train_val_iterator_from_tfr_in_local(
                input_path, batch_size, tfr_feature_key, n_feature,
                n_row, pre_fix, data_cache)
    return [train_iter, val_iter, batchs_train, batchs_val]


def decode2(serialized_example, id_key, datetime_key, feature_key, n_input=None):
    unserialized = tf.parse_single_example(
        serialized_example,
        features={id_key: tf.FixedLenFeature([], dtype=tf.string),
                  datetime_key: tf.FixedLenFeature([], dtype=tf.string),
                  feature_key: tf.FixedLenFeature([n_input], dtype=tf.float32)
                  })
    return unserialized


def get_iterator_from_tfr_in_s3(
        folder_path, batch_size, tfr_id_key, tfr_datetime_key,
        tfr_feature_key, n_input, n_rows, pre_fix, is_train=False, cache=False):
    print("folder_path:", folder_path)
    result = re.search("s3:\/\/(.+?)\/(.*)", folder_path)
    bucket_name = result.group(1)
    tfr_path = result.group(2)
    client = boto3.client('s3')
    response = client.list_objects(Bucket=bucket_name,
                                   Prefix=os.path.join(tfr_path, pre_fix))
    # print "response:", response
    filenames = [os.path.join('s3://', bucket_name, item['Key']) for item in response['Contents']]
    print("filenames length:", len(filenames))
    print(filenames[:1])

    DATASET_SIZE = round(n_rows / 100)
    batchs_per_train = int(DATASET_SIZE / batch_size) + 1
    full_dataset = tf.data.TFRecordDataset(filenames[:2], num_parallel_reads=8)
    full_dataset = full_dataset.map(
        lambda x: decode2(
            x,
            tfr_id_key,
            tfr_datetime_key,
            tfr_feature_key,
            n_input=n_input),
        num_parallel_calls=8)

    if is_train:
        full_dataset = full_dataset.shuffle(1000 + 3 * batch_size)
        if cache:
            full_dataset = full_dataset.cache()
        full_dataset = full_dataset.repeat()
    full_dataset = full_dataset.batch(batch_size)
    full_dataset = full_dataset.prefetch(4)
    iterator = full_dataset.make_one_shot_iterator()
    return [iterator, batchs_per_train]


def get_iterator_from_dataframe(dataframe, batch_size=64, cache=False):
    print("DATASET_SIZE: ", dataframe.shape)
    batchs_per_epoch = int(dataframe.shape[0] / batch_size) + 1
    dataset = tf.data.Dataset.from_tensor_slices(dataframe)
    dataset = dataset.shuffle(buffer_size=128)
    #     dataset = dataset.map(lambda x:_parse_and_preprocess(x))
    if cache:
        dataset = dataset.cache()
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(4)

    batch_iterator = dataset.make_one_shot_iterator()
    return batch_iterator, batchs_per_epoch


def get_train_val_iterator_from_dataframe(dataframe, batch_size, cache=False):
    print("DATASET_SIZE: ", dataframe.shape)
    # batchs_per_epoch = features.shape[0]/batch_size
    full_dataset = tf.data.Dataset.from_tensor_slices(dataframe)
    full_dataset = full_dataset.shuffle(10 * batch_size + 1000)
    DATASET_SIZE = dataframe.shape[0]
    train_size = int(0.95 * DATASET_SIZE)
    val_size = int(0.05 * DATASET_SIZE)
    print("train_size: ", train_size, "val_size: ", val_size)
    batchs_per_train = int(train_size / batch_size) + 1
    batchs_per_val = int(val_size / batch_size) + 1
    train_dataset = full_dataset.take(train_size)
    val_dataset = full_dataset.skip(val_size)

    if cache:
        train_dataset = train_dataset.cache()
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(1)
    train_iterator = train_dataset.make_one_shot_iterator()

    if cache:
        val_dataset = val_dataset.cache()
    val_dataset = val_dataset.repeat()
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.prefetch(1)
    val_iterator = val_dataset.make_one_shot_iterator()

    return [train_iterator, val_iterator, batchs_per_train, batchs_per_val]


def save_json(json_object, folder_path, json_name):
    if 's3://' in folder_path:
        result = re.search("s3:\/\/(.+?)\/(.*)", folder_path)
        bucket_name = result.group(1)
        file_path = result.group(2)
        obj = s3.Object(bucket_name, os.path.join(file_path, json_name))
        obj.put(Body=json.dumps(json_object))
    else:
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        with open(os.path.join(folder_path, json_name), 'w') as outfile:
            json.dump(json_object, outfile)


def load_json(folder_path, json_name):
    if "s3://" in folder_path:
        result = re.search("s3:\/\/(.+?)\/(.*)", folder_path)
        bucket_name = result.group(1)
        file_path = result.group(2)
        s3 = boto3.resource('s3')
        content_object = s3.Object(
            bucket_name=bucket_name,
            key=os.path.join(file_path, json_name))
        file_content = content_object.get()['Body'].read().decode('utf-8')
        json_object = json.loads(file_content)
    else:
        with open(os.path.join(folder_path, json_name), "r") as read_file:
            json_object = json.load(read_file)
    return json_object


def get_tfr_json_from_s3(bucket_name, json_path):
    try:
        s3 = boto3.resource('s3')
        content_object = s3.Object(bucket_name=bucket_name, key=json_path)
        file_content = content_object.get()['Body'].read().decode('utf-8')
        json_content = json.loads(file_content)
        tfr_json = json_content['tfrecords_info']
        return tfr_json

    except Exception as e:
        print("================ ERROR ================")
        print("couldn't access S3 with the following reason.")
        print(e)
        return None


def save_df_to_csv_in_s3(dataframe, save_path, csv_name):
    result = re.search("s3:\/\/(.+?)\/(.*)", save_path)
    bucket_name = result.group(1)
    file_path = result.group(2)
    csv_buffer = BytesIO()
    dataframe.to_csv(csv_buffer)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket_name, os.path.join(file_path, csv_name)) \
        .put(Body=csv_buffer.getvalue())
