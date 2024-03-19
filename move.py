import json
import shutil
import os
from glob import glob
from tqdm import tqdm
from config import config
try:
    for i in range(0,8):
        os.mkdir(config.train_data + str(i))
except:
    pass
    
file_train = json.load(open(config.train_json,"r",encoding="utf-8"))
file_val = json.load(open(config.valid_json,"r",encoding="utf-8"))

file_list = file_train + file_val

for file in tqdm(file_list):
    filename = file["image_id"]
    origin_path = config.root + "/temp/images/" + filename
    ids = file["disease_class"]
    if ids ==  44:
        continue
    if ids == 45:
        continue
    if ids > 45:
        ids = ids -2
    save_path = config.train_data + str(ids) + "/"
    shutil.copy(origin_path,save_path)

