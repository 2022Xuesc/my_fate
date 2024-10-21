import numpy as np
import cv2
from PIL import Image
import os

image_dir = '/home/klaus125/research/dataset/NUS-WIDE/images'

types = ['val']

for type in types:
    image_type_dir = os.path.join(image_dir,type)
    cnt = 0
    files = os.listdir(image_type_dir)
    total = len(files)
    bad_cnt = 0
    for filename in files:
        image_path = os.path.join(image_type_dir, filename)
        data = cv2.imread(image_path)
        cnt += 1
        # print(f'{cnt} / {total}')
        if data is None:
            print(f'{filename} is bad.')
            bad_cnt += 1
    print(bad_cnt)
    
