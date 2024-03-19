# https://github.com/spytensor/plants_disease_detection
#
#

import os

import random
import time
import json
import torch
import torchvision
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from torch import nn, optim
from config import config
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataloader import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from timeit import default_timer as timer
from models.model import *
from utils import *
from flask import Flask
# 1. set random.seed and cudnn performance
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')


# 2. evaluate func



# 3. test model on public dataset and save the probability matrix
def test(test_loader, model, folds):
    # 3.1 confirm the model converted to cuda
    csv_map = OrderedDict({"filename": [], "probability": []})
    model.to(device)
    model.eval()
    with open("./submit/baseline.json", "w", encoding="utf-8") as f:
        submit_results = []
        for i, (input, filepath) in enumerate(tqdm(test_loader)):
            # 3.2 change everything to cuda and get only basename
            filepath = [os.path.basename(x) for x in filepath]
            with torch.no_grad():
                image_var = Variable(input).to(device)
                # 3.3.output
                # print(filepath)
                # print(input,input.shape)
                y_pred = model(image_var)
                # print(y_pred.shape)
                smax = nn.Softmax(1)
                smax_out = smax(y_pred)
            # 3.4 save probability to csv files
            csv_map["filename"].extend(filepath)
            for output in smax_out:
                prob = ";".join([str(i) for i in output.data.tolist()])
                csv_map["probability"].append(prob)
        result = pd.DataFrame(csv_map)
        result["probability"] = result["probability"].map(lambda x: [float(i) for i in x.split(";")])
        for index, row in result.iterrows():
            pred_label = np.argmax(row['probability'])
            if pred_label > 43:
                pred_label = pred_label + 2
            submit_results.append({"image_id": row['filename'], "disease_class": pred_label})
        json.dump(submit_results, f, ensure_ascii=False, cls=MyEncoder)


# 4. more details to build main function
def main():

    fold = config.fold
    # 4.1 mkdirs

    if not os.path.exists(config.submit):
        os.mkdir(config.submit)
    if not os.path.exists(config.weights):
        os.mkdir(config.weights)
    if not os.path.exists(config.best_models):
        os.mkdir(config.best_models)
    if not os.path.exists(config.logs):
        os.mkdir(config.logs)
    if not os.path.exists(config.weights + config.model_name + os.sep + str(fold) + os.sep):
        os.makedirs(config.weights + config.model_name + os.sep + str(fold) + os.sep)
    if not os.path.exists(config.best_models + config.model_name + os.sep + str(fold) + os.sep):
        os.makedirs(config.best_models + config.model_name + os.sep + str(fold) + os.sep)
        # 4.2 get model and optimizer
    model = get_net()
    # model = torch.nn.DataParallel(model)
    model.to(device)
    # optimizer = optim.SGD(model.parameters(),lr = config.lr,momentum=0.9,weight_decay=config.weight_decay)
    # criterion = FocalLoss().cuda()


    # 4.4 restart the training process

    test_files = get_files(config.test_data, "test")

    """ 
    #4.5.2 split
    split_fold = StratifiedKFold(n_splits=3)
    folds_indexes = split_fold.split(X=origin_files["filename"],y=origin_files["label"])
    folds_indexes = np.array(list(folds_indexes))
    fold_index = folds_indexes[fold]

    #4.5.3 using fold index to split for train data and val data
    train_data_list = pd.concat([origin_files["filename"][fold_index[0]],origin_files["label"][fold_index[0]]],axis=1)
    val_data_list = pd.concat([origin_files["filename"][fold_index[1]],origin_files["label"][fold_index[1]]],axis=1)
    """

    test_dataloader = DataLoader(ChaojieDataset(test_files, test=True), batch_size=1, shuffle=False, pin_memory=False)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,"max",verbose=1,patience=3)
    load_best_path = config.best_models + os.sep + config.model_name + os.sep + str(fold) + os.sep + 'model_best.pth.tar'
    load_epoch_path = config.weights + config.model_name + os.sep + str(fold) + os.sep + 'epoch3.pth.tar'
    print(os.path.abspath(load_epoch_path))
    best_model = torch.load(
        load_epoch_path, map_location=torch.device('cpu')
    )
    model.load_state_dict(best_model["state_dict"])
    test(test_dataloader, model, fold)


if __name__ == "__main__":
    device = torch.device("cuda:" + config.gpus)
    main()








































