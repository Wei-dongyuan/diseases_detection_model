# https://github.com/spytensor/plants_disease_detection
#
#
from fractions import Fraction
import os
import random
import subprocess
import time
import json
import torch
import torchvision
import numpy as np
import pandas as pd
import os
import os  
import glob  
import warnings
import ultralytics
from mRandom import generate_random_disease_probabilities
from mRandom import generate_random_pest_probabilities,generate_random_pest_quantities
from datetime import datetime
from torch import nn, optim
from config import config
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataloader import *
from ultralytics1main.predict_one import predict_one
from sklearn.model_selection import train_test_split, StratifiedKFold
from timeit import default_timer as timer
from models.model import *
from utils import *
from flask import Flask, jsonify, request
from gevent import pywsgi
from ultralytics import YOLO
from download_image import *
from PIL import Image, TiffImagePlugin,ExifTags
from PIL.ExifTags import TAGS, GPSTAGS

baseUrl = "http://127.0.0.1:9000/"
save_directory = "D:\code\program\Diseaseimage"
model_count = YOLO('yolov8n.yaml').load('./ultralytics1main/best.pt')  # 从YAML构建并转移权重
# print(model_count.type)
# predict_one('img1.jpg', model_count)
app = Flask(__name__)


device = torch.device("cpu")
    

config.num_classes = 8
model_disease = get_net()
model_disease.to(device)
load_best_path = config.best_models + os.sep + config.model_name + os.sep + str('apple') + os.sep + 'model_best.pth.tar'
best_model = torch.load(
    load_best_path, map_location=torch.device('cpu')
)
model_disease.load_state_dict(best_model["state_dict"])
model_disease.to(device)
model_disease.eval()



config.num_classes = 10
config.batch_size = 1
model_pest = get_net()
model_pest.to(device)
load_best_path = config.best_models + os.sep + config.model_name + os.sep + str('insect') + os.sep + 'model_best.pth.tar'
best_model = torch.load(
    load_best_path, map_location=torch.device('cpu')
)
model_pest.load_state_dict(best_model["state_dict"])
model_pest.to(device)
model_pest.eval()
dis_dic = ['苹果腐烂病','褐斑病','白粉锈病','炭疽叶枯病','白粉病','炭疽病','苹果霉心病','斑点落叶病']
pest_dic = ['绵蚜','越冬虫卵','绿盲蝽','金龟子','红蜘蛛','苹果食心虫','卷叶虫','蚜虫','金蚊细蛾','食心虫']
def pest_detection(img, model):
    # 3.1 confirm the model converted to cuda
    test_files = get_one_file(img)
    test_dataloader = DataLoader(ChaojieDataset(test_files, test=True), batch_size=1, shuffle=False, pin_memory=False)
    # with open("./submit/baseline.json", "w", encoding="utf-8") as f:
    for i, (input, filepath) in enumerate(test_dataloader):
        with torch.no_grad():
            image_var = Variable(input).to(device)
            y_pred = model(image_var)
            y_pred = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
            y_pred = y_pred / torch.sum(y_pred)
            y_pred = y_pred.tolist()
            return dict(zip(pest_dic, y_pred[0]))
        
def disease_detection(img, model):
    config.num_classes = 10
    # 3.1 confirm the model converted to cuda
    test_files = get_one_file(img)
    test_dataloader = DataLoader(ChaojieDataset(test_files, test=True), batch_size=1, shuffle=False, pin_memory=False)
    # with open("./submit/baseline.json", "w", encoding="utf-8") as f:
    for i, (input, filepath) in enumerate(test_dataloader):
        with torch.no_grad():
            image_var = Variable(input).to(device)
            y_pred = model(image_var)
            print(y_pred)
            y_pred = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
            y_pred = y_pred / torch.sum(y_pred)
            y_pred = y_pred.tolist()
            return dict(zip(dis_dic, y_pred[0]))

def get_one_file(root):
    #for test
    files = []
    files.append(root)
    files = pd.DataFrame({"filename":files})
    return files

# 4. more details to build main function
# @app.route('/test', methods=["POST"]) 
# def test():
#     path=request.form.get('path')
#     resmsg=pest_detection(path, model_pest)
#     print(resmsg)
#     return resmsg

@app.route('/detectPestImage', methods=["POST"])    
def detectPestImage():
    url = request.form.get('path')
    path = download_image(baseUrl+url,save_directory)
    resmsg=pest_detection(path, model_pest)
    delete_specific_image(path)
    print(url)
    print(resmsg)
    # res = generate_random_pest_probabilities()
    # print(res)
    return resmsg

@app.route('/detectPestAndNum', methods=["POST"])    
def detectPestAndNum():
    url = request.form.get('path')
    path = download_image(baseUrl+url,save_directory)
    resmsg=predict_one(path, model_count)
    delete_specific_image(path)
    print(url)
    # resmsg = { "0":1, "1":2, "2":0,"3":4}
    # res = generate_random_pest_quantities()
    print(resmsg)
    return resmsg
# @app.route('/detectPestNum', methods=["Get"])
# def detectPestsNum():
#     path = request.form.get('path')
#     print("hello 8")
#     res = 8
#     return res
def rational_to_float(rational):
    return float(Fraction(rational[0], rational[1])) if rational and rational[1] != 0 else None
def extract_gps_info(gps_info):
    # gps_info = exif_data.get(34853, {})  # 34853 is the tag for GPSInfo
    latitude = gps_info.get(2)  # GPSLatitude
    longitude = gps_info.get(4)  # GPSLongitude

    # 如果是 tuple 类型，转换为浮点数
    if isinstance(latitude, tuple) and isinstance(longitude, tuple):
        latitude = rational_to_float(latitude)
        longitude = rational_to_float(longitude)

    return {'latitude': latitude, 'longitude': longitude}
def get_gps_of_TIFF(exif_data):
    if exif_data is not None:
                # 提取纬度和经度信息
        latitude = None
        longitude = None
        # print(exif_data)
                
        for tag, value in exif_data.items():  
            # if tag == 34853:
            #     gps_info  = extract_gps_info(value)
            #     return {'exif_data': gps_info}
            if tag == 0x8825:  # GPSInfo tag
                if isinstance(value, dict):
                    for key, sub_value in value.items():
                        sub_tag = GPSTAGS.get(key, key)
                        if sub_tag == 'GPSLatitude':
                            latitude = sub_value
                        elif sub_tag == 'GPSLongitude':
                            longitude = sub_value


            if tag == 700 and isinstance(value, bytes):
                metadata_str = value.decode('utf-8')
                if 'drone-dji:GpsLatitude' in metadata_str and 'drone-dji:GpsLongitude' in metadata_str:
                    latitude_str = metadata_str.split('drone-dji:GpsLatitude="')[1].split('"')[0]
                    longitude_str = metadata_str.split('drone-dji:GpsLongitude="')[1].split('"')[0]

                    latitude = float(latitude_str)
                    longitude = float(longitude_str)
        print({'latitude': latitude, 'longitude': longitude})
        return {'latitude': latitude, 'longitude': longitude}
def get_gps_of_JPG(exif_data):
    latitude = None
    longitude = None

    if 34853 in exif_data:
        gps_info = exif_data[34853]
        print(gps_info[2])
        print(gps_info[4])
        latitude = convert_degrees(gps_info[2]) if 2 in gps_info else None
        longitude = convert_degrees(gps_info[4]) if 4 in gps_info else None

    return {'latitude': latitude, 'longitude': longitude}

def convert_degrees(gps_data):
    if not gps_data:
        return None

    degrees = float(gps_data[0]) 
    minutes = float(gps_data[1]) 
    seconds = float(gps_data[2]) 
    
    return degrees + minutes / 60 + seconds / 3600

def get_exif_data(image_path):
    try:
        with Image.open(image_path, 'r') as img:
            # 如果是 TIFF 文件，使用 img.tag_v2
            if img.format == 'TIFF':
                exif_data = img.tag_v2
                return get_gps_of_TIFF(exif_data)
                
            else:
                exif_data = img._getexif()
                return get_gps_of_JPG(exif_data)
    except Exception as e:
        print(f"Error extracting EXIF data: {str(e)}")
    return None

@app.route('/getImageCoordinates', methods=['POST'])
def getImageCoordinates():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file:
            # 保存上传的文件
            file_path = 'uploaded_image.jpg'
            file.save(file_path)

            # 获取经纬度信息
            exif_data = get_exif_data(file_path)

            return exif_data

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/detectDiseaseImage",methods=["POST"])     
def detectDiseaseImage():
    url = request.form.get('path')
    print("url"+url)
    path = download_image(baseUrl+url,save_directory)
    print(url)
    resmsg=disease_detection(path, model_disease)
    print(resmsg)
    delete_specific_image(path)
    # res = generate_random_disease_probabilities()
    # print(res)
    return resmsg

@app.route("/hellomsg",methods=["POST"])    
def hello():
    msg = request.form.get('msg')
    
    return msg

@app.route("/hello",methods=['GET'])    
def hellotest():
    return "hello world"

def print_files_in_directory(directory_path):
    try:
        # 获取目录下的所有文件
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

        # 直接输出文件列表
        for file_name in files:
            print(file_name)

    except Exception as e:
        print(f"Error: {str(e)}")

@app.route('/execute_script', methods=['GET'])
def execute_script():
    try:
        # 手动指定虚拟环境和脚本路径
        conda_env = 'pytorch'
        script_path1 = 'D:\code\program\smart-garden-service\\src\\resources\\UAVImage\\UAV-image-stitching.py'
        script_path2 = 'D:\code\program\smart-garden-service\\src\\resources\\UAVImage\\Untitled-2.py'

        # 构建激活 conda 虚拟环境的命令
        activate_cmd = f'conda activate {conda_env} &&'
        print("test 1")
        print_files_in_directory('D:\code\program\smart-garden-service\\src\\resources\\UAVImage')
        # 构建执行脚本的命令
        script_cmd1 = f'python {script_path1}'
        result1 = subprocess.run(f'{activate_cmd} {script_cmd1}', shell=True, capture_output=True, text=True)
        print_files_in_directory('D:\code\program\smart-garden-service\\src\\resources\\UAVImage')
        print("Return code:", result1.returncode)
        print("Standard Output:", result1.stdout)
        print("Standard Error:", result1.stderr)
        # 执行命令
        if result1.returncode == 0:
            # print("test 3")
            # script_cmd2 = f'python {script_path2}'
            # result2 = subprocess.run(f'{activate_cmd} {script_cmd2}', shell=True, capture_output=True, text=True)
            print("test 5")
            return jsonify({'output': result1.stdout, 'error': result1.stderr}), 200
        else:
            print("test 4")
            return jsonify({'error': result1.stderr}), 500
            # 返回执行结果
            
        # 返回执行结果

    except Exception as e:
        print("failed 1")
        return jsonify({'error': str(e)}), 500
    
@app.route("/detectMultispectralDiseaseImage",methods=["POST"])    
def detectMultispectralDiseaseImage():
    return generate_random_disease_probabilities()
if __name__ == "__main__":
    server = pywsgi.WSGIServer(('0.0.0.0',5000),app)
    server.serve_forever()
    app.run()
    # img = 'C:\\Users\\DELL\\Desktop\\a.jpg'
    # a = predict_one(img, model_count)
    # print(a)

































