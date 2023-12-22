import json
import os.path
import pickle
import numpy as np


def get(vec):
    return np.array([float(num) for num in vec], ).astype(np.float32)


category_file_name = 'category.json'
dataset = 'voc_expanded'
data_dir = f'/home/klaus125/research/fate/my_practice/dataset/{dataset}'
categories = json.load(open(os.path.join(data_dir, category_file_name), 'r'))
word2vec = json.load(open('../glove/word2vec.json', 'r'))

compound_words = {'diningtable': ['dining', 'table'], 'pottedplant': ['potted', 'plant'],
                  'tvmonitor': ['tv', 'monitor']}
num_classes = len(categories)
matrix = np.ndarray((num_classes, 300)).astype(np.float32)
for category in categories:
    if category in compound_words:
        vec = np.zeros(300)
        for part_category in compound_words[category]:
            vec += get(word2vec[part_category])
        vec /= len(compound_words[category])
    else:
        vec = get(word2vec[category])
    matrix[categories[category]] = vec
# 将matrix保存为pkl
pkl_name = f'{dataset}_glove_word2vec.pkl'
pickle_output = open(os.path.join(data_dir, pkl_name), 'wb')
pickle.dump(matrix, pickle_output)


def check_coco_and_nuswide():
    # Todo: 验证之前数据集的vec是否正确
    dataset = 'nuswide'
    # dataset = 'coco'

    data_dir = f'/home/klaus125/research/fate/my_practice/dataset/{dataset}'
    pkl_file_name = f'{dataset}_glove_word2vec.pkl'
    category_file_name = 'category.json'
    pkl_data = pickle.load(open(os.path.join(data_dir, pkl_file_name), 'rb'))
    categories = json.load(open(os.path.join(data_dir, category_file_name), 'r'))

    word2vec = json.load(open('../glove/word2vec.json', 'r'))

    # 遍历每个category，判断对应的vec和pkl中的是否一致
    for category in categories:
        if category not in word2vec:
            continue
        print(
            np.array([float(num) for num in word2vec[category]], ).astype(np.float32) == pkl_data[categories[category]])
