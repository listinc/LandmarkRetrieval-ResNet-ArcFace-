import os
import sys
import pickle
import shutil
import csv
import time
import multiprocessing

import numpy as np
import pandas as pd

from scipy.spatial import distance
from Resnet import dataset

from multiprocessing import Pool
from tqdm import tqdm


def perform_search(queryFeatures, index, count):
    result = distance.cdist([queryFeatures], index["features"], 'euclidean')

    results = list(map(lambda x, y: (x, y), result[0], index["id"]))
    results = sorted(results)[:count]

    predicted = [r[1] for r in results]

    return predicted


def test_mAP(i):
    query_filename = str(query['id'][i]).split('/')[1]
    query_class = str(query['id'][i]).split('/')[0]

    indexes = []
    for j in range(0, len(index['id'])):
        if query_class in str(index['id'][j]):
            indexes.append(str(index['id'][j]))

    predicted = perform_search(query['features'][i], index, retrieval_count)

    index_ids = []
    ids_num = 1
    append = index_ids.append
    for pre_label in predicted:
        if pre_label in indexes:
            for im_idx, im_label in enumerate(indexes):
                if pre_label == im_label:
                    append(im_idx)
        else:
            append(len(indexes) + ids_num)
            ids_num += 1

    #     print(str(i)+'/'+str(len(query['id'])))
    #     print(str(i)+'/'+str(len(query['id'])), query_filename, predicted)

    ok_ids = list(range(len(indexes)))
    sorted_index_ids.append(index_ids)
    ground_truth.append({'ok': np.array(ok_ids), 'junk': np.array([], dtype='int64')})

    return (sorted_index_ids, ground_truth)


start = time.time()

DATA_PATH = '/home/list-pc31/hdd/hdd_ext/2020_08_LANDMARK/LANDMARK_Dataset/'
result_path = 'train_result/3/'

index = pickle.loads(open(result_path + 'index_feature.pickle', "rb").read())
query = pickle.loads(open(result_path + 'query_feature.pickle', "rb").read())

# query_classes = []
query_classes = [str(query["id"][i]).split('/')[0] for i in range(0, len(query["id"]))]
# for i in range(0, len(query["id"])):
#     query_class = str(query["id"][i]).split('/')[0]
#     query_classes.append(query_class)
query_classes = set(query_classes)

sorted_index_ids = []
ground_truth = []

retrieval_count = 50

# query_idx = list(range(len(query["features"])))
query_idx = list(range(85000, 90000))
with Pool(10) as p:
    res = list(tqdm(p.imap(test_mAP, query_idx), total=len(query_idx)))

res_np = np.asarray(res)
print(res_np.shape)
np.save(result_path+'query_result_np'+str(90000), res_np)

for res_i in res:
    sorted_index_ids.append(res_i[0][0])
    ground_truth.append(res_i[1][0])
#     ground_truth.append(ground_truth.append({'ok': np.array(res_i[1]), 'junk': np.array([], dtype='int64')}))

print(time.time() - start)

(mean_average_precision, mean_precisions, mean_recalls, average_precisions, precisions, recalls) \
    = dataset.ComputeMetrics(np.array(sorted_index_ids), ground_truth, [retrieval_count])
print(mean_average_precision)

for sii_idx, sii in enumerate(sorted_index_ids):
    sorted_index_ids[sii_idx] = sii[:20]
(mean_average_precision, mean_precisions, mean_recalls, average_precisions, precisions, recalls) \
    = dataset.ComputeMetrics(np.array(sorted_index_ids), ground_truth, [20])
print(mean_average_precision)

for sii_idx, sii in enumerate(sorted_index_ids):
    sorted_index_ids[sii_idx] = sii[:15]
(mean_average_precision, mean_precisions, mean_recalls, average_precisions, precisions, recalls) \
    = dataset.ComputeMetrics(np.array(sorted_index_ids), ground_truth, [15])
print(mean_average_precision)

for sii_idx, sii in enumerate(sorted_index_ids):
    sorted_index_ids[sii_idx] = sii[:10]
(mean_average_precision, mean_precisions, mean_recalls, average_precisions, precisions, recalls) \
    = dataset.ComputeMetrics(np.array(sorted_index_ids), ground_truth, [10])
print(mean_average_precision)

for sii_idx, sii in enumerate(sorted_index_ids):
    sorted_index_ids[sii_idx] = sii[:5]
(mean_average_precision, mean_precisions, mean_recalls, average_precisions, precisions, recalls) \
    = dataset.ComputeMetrics(np.array(sorted_index_ids), ground_truth, [5])
print(mean_average_precision)

print(time.time() - start)

# import os
# import sys
# import pickle
# import shutil
# import csv
# import time
#
# import numpy as np
# import pandas as pd
#
# from scipy.spatial import distance
# from Resnet import dataset
#
#
# def perform_search(queryFeatures, index):
#     results = []
#     predicted = []
#     score = []
#     # for i in range(0, len(index["features"])):
#     #     d = distance.euclidean(queryFeatures, index["features"][i])
#     #     id = str(index["id"][i]).split('/')[1]
#     #     results.append((d, id))
#     result = distance.cdist([queryFeatures], index["features"], 'euclidean')
#     results = list(map(lambda x, y: (x, y), result[0], index["id"]))
#
#     return results
#
# #     results = sorted(results)[:maxResults]
# #     for r in results:
# #         predicted.append(r[1])
# #         score.append(r[0])
#
# #     return score, predicted
#
#
# def return_result(predicted, indexes):
#     index_ids = []
#     ids_num = 1
#     for pre_label in predicted:
#         if pre_label in indexes:
#             for im_idx, im_label in enumerate(indexes):
#                 if pre_label == im_label:
#                     index_ids.append(im_idx)
#         else:
#             index_ids.append(len(indexes) + ids_num)
#             ids_num += 1
#     return index_ids
#
#
# start = time.time()
#
# retrieval_counts = [5, 10, 15, 20, 50]
#
# DATA_PATH = '/home/list-pc31/hdd/hdd_ext/2020_08_LANDMARK/LANDMARK_Dataset/'
# result_path = 'train_result/3/'
#
# index = pickle.loads(open(result_path + 'index_feature.pickle', "rb").read())
# query = pickle.loads(open(result_path + 'query_feature.pickle', "rb").read())
#
# # f = open(result_path+'result_'+str(retrieval_count)+'.csv', 'w', newline='', encoding='utf-8')
# # wr = csv.writer(f)
#
# query_classes = []
# for i in range(0, len(query["id"])):
#     query_class = str(query["id"][i]).split('/')[0]
#     query_classes.append(query_class)
# query_classes = set(query_classes)
#
# sorted_index_ids_5 = []
# sorted_index_ids_10 = []
# sorted_index_ids_15 = []
# sorted_index_ids_20 = []
# sorted_index_ids_50 = []
# ground_truth = []
# for i in range(0, len(query["features"])):
#     query_filename = str(query['id'][i]).split('/')[1]
#     query_class = str(query['id'][i]).split('/')[0]
#
#     indexes = []
#     for j in range(0, len(index['id'])):
#         if query_class in str(index['id'][j]):
#             indexes.append(str(index['id'][j]))
#
#     #     predicted_score, predicted = perform_search(
#     #         query['features'][i],
#     #         index,
#     #         maxResults=retrieval_count)
#     results = perform_search(
#         query['features'][i],
#         index)
#
#     for rc in retrieval_counts:
#         predicted = []
#         sorted_results = sorted(results)[:rc]
#         for r in sorted_results:
#             predicted.append(r[1])
#         #         wr.writerow([query_filename, predicted])
#
#         index_ids = return_result(predicted, indexes)
#         if rc == 5:
#             sorted_index_ids_5.append(index_ids)
#             print(str(i) + '/' + str(len(query['id'])), query_filename, predicted)
#         elif rc == 10:
#             sorted_index_ids_10.append(index_ids)
#         elif rc == 15:
#             sorted_index_ids_15.append(index_ids)
#         elif rc == 20:
#             sorted_index_ids_20.append(index_ids)
#         elif rc == 50:
#             sorted_index_ids_50.append(index_ids)
#
#     ok_ids = list(range(len(indexes)))
#     ground_truth.append({'ok': np.array(ok_ids), 'junk': np.array([], dtype='int64')})
#
# # f.close()
#
# (mean_average_precision, mean_precisions, mean_recalls, average_precisions, precisions, recalls) \
#     = dataset.ComputeMetrics(np.array(sorted_index_ids_5), ground_truth, [5])
# print(5)
# print(mean_average_precision)
#
# (mean_average_precision, mean_precisions, mean_recalls, average_precisions, precisions, recalls) \
#     = dataset.ComputeMetrics(np.array(sorted_index_ids_10), ground_truth, [10])
# print(10)
# print(mean_average_precision)
#
# (mean_average_precision, mean_precisions, mean_recalls, average_precisions, precisions, recalls) \
#     = dataset.ComputeMetrics(np.array(sorted_index_ids_15), ground_truth, [15])
# print(15)
# print(mean_average_precision)
#
# (mean_average_precision, mean_precisions, mean_recalls, average_precisions, precisions, recalls) \
#     = dataset.ComputeMetrics(np.array(sorted_index_ids_20), ground_truth, [20])
# print(20)
# print(mean_average_precision)
#
# (mean_average_precision, mean_precisions, mean_recalls, average_precisions, precisions, recalls) \
#     = dataset.ComputeMetrics(np.array(sorted_index_ids_50), ground_truth, [50])
# print(50)
# print(mean_average_precision)
#
# print(time.time() - start)
