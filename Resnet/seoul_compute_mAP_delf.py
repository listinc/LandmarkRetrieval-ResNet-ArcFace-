import os
import sys
import pickle
import shutil
import csv

import numpy as np
import pandas as pd

import ml_metrics as ml

from scipy.spatial import distance
from Resnet import dataset


def perform_search(queryFeatures, index, maxResults=100):
    results = []
    predicted = []
    score = []
    for i in range(0, len(index["features"])):
        d = distance.euclidean(queryFeatures, index["features"][i])
        id = str(index["id"][i]).split('/')[1]
        results.append((d, id))

    results = sorted(results)[:maxResults]
    for r in results:
        predicted.append(r[1])
        score.append(r[0])

    return score, predicted


retrieval_count = 20

DATA_PATH = '/home/list-pc31/hdd/hdd_ext/2020_08_LANDMARK/LANDMARK_Dataset/'
data_path = '1차_2차_합/서울시_split/'

train_path = 'train_result/seoul_1and2_test3/'
index = pickle.loads(open(train_path+'index.pickle', "rb").read())
query = pickle.loads(open(train_path+'test.pickle', "rb").read())

train_img_path = os.path.join(DATA_PATH, data_path+'train')
val_img_path = os.path.join(DATA_PATH, data_path+'val')
test_img_path = os.path.join(DATA_PATH, data_path+'query')

landmarks = os.listdir(test_img_path)

sorted_index_ids = []
ground_truth = []
for i in range(0, len(query["features"])):
    query_filename = str(query['id'][i]).split('/')[1]
    query_class = ''

    for l in landmarks:
        ims = os.listdir(test_img_path + '/' + l)
        if query_filename in ims:
            query_class = l

    train_images = os.listdir(train_img_path + '/' + query_class)
    val_images = os.listdir(val_img_path + '/' + query_class)
    index_images = train_images + val_images

    predicted_score, predicted = perform_search(query['features'][i], index, maxResults=retrieval_count)
    print(query_filename, predicted)
    
    ok_ids = list(range(len(index_images)))

    index_ids = []
    ids_num = 1
    for pre_label in predicted:
        if pre_label in index_images:
            for im_idx, im_label in enumerate(index_images):
                if pre_label == im_label:
                    index_ids.append(im_idx)
        else:
            index_ids.append(len(index_images)+ids_num)
            ids_num += 1

    sorted_index_ids.append(index_ids)
    ground_truth.append({'ok': np.array(ok_ids), 'junk': np.array([], dtype='int64')})

(mean_average_precision, mean_precisions, mean_recalls, average_precisions, precisions, recalls) \
    = dataset.ComputeMetrics(np.array(sorted_index_ids), ground_truth, [retrieval_count])

print(mean_average_precision)
