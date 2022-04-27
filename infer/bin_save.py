import json
import os
import numpy as np
import cv2
import shutil
from tqdm import tqdm

with open('model.json', 'r') as f:
    bbox_dic = json.load(f)

for i in range(10):
    if not os.path.exists('./img_bin/{}/'.format(i)):
        os.makedirs('./img_bin/{}/'.format(i))

for k, v in tqdm(bbox_dic.items()):
    i = int((v * 10) / 1)
    shutil.copy(k, './img_bin/{}/'.format(i))
