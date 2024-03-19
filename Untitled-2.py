import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import random
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def split_image(image, input):
    height = input
    width = input
    image_height, image_width, _ = image.shape
    num_horizontal = image_width // width
    num_vertical = image_height // height
    image_list = [[None] * num_horizontal for _ in range(num_vertical)]

    for i in range(num_vertical):
        for j in range(num_horizontal):
            left = j * width
            upper = i * height
            right = left + width
            lower = upper + height
            cropped_image = image[upper:lower, left:right]
            image_list[i][j] = cropped_image

    return image_list

def image_to_vector(image):
    image_array = np.array(image)
    image_vector = image_array.flatten()
    
    return image_vector

    
    return result
def pca(vector, num_components):
    covariance_matrix = np.cov(vector.T)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    selected_indices = sorted_indices[:num_components]
    principal_components = eigenvectors[:, selected_indices]
    reduced_vector = np.dot(vector, principal_components)
    
    return reduced_vector

def randomPredict(vector):
    return np.array([random.random() for _ in range(vector.shape[0])])

def detector():
    width = 100
    img_list_raw = []
    img_list_raw.append(cv2.imread("result_R_FINAL.TIF"))
    img_list_raw.append(cv2.imread("result_G_FINAL.TIF"))
    img_list_raw.append(cv2.imread("result_RE_FINAL.TIF"))
    img_list_raw.append(cv2.imread("result_NIR_FINAL.TIF"))
    img_RGB = cv2.imread("result_RGB_FINAL.JPG")

    img_RGB_list = split_image(img_RGB, width)
    img_all_list = []
    image_feature_array = []
    image_reduced_array = []
    for loop in range(4):
        img_all_list.append(split_image(img_list_raw[loop], width))
    row_num = len(img_all_list[0])
    column_num = len(img_all_list[0][0])
    print(row_num)
    print(column_num)
    image_feature_array_list = [[],[],[],[]]  
    for loop in range(row_num):
        for loop1 in range(column_num):
            image_feature_array_list[0].append(image_to_vector(img_all_list[0][loop][loop1]).reshape(3 * width * width,1))
            image_feature_array_list[1].append(image_to_vector(img_all_list[1][loop][loop1]).reshape(3 * width * width,1))
            image_feature_array_list[2].append(image_to_vector(img_all_list[2][loop][loop1]).reshape(3 * width * width,1))
            image_feature_array_list[3].append(image_to_vector(img_all_list[3][loop][loop1]).reshape(3 * width * width,1))


    image_feature_list = []
    image_reduced_list = []
    for loop in range(4):
        image_feature_list.append(np.concatenate(image_feature_array_list[loop], axis = 1))
        print(image_feature_list[loop].shape)
        image_reduced_list.append(pca(image_feature_list[loop].T, 10).T)
    
    image_feature_array = np.concatenate(image_reduced_list)
    # print(image_feature_array.shape)
    # classifier = svm.SVC(kernel='linear')
    # y = classifier.predict(image_feature_array.T)
    y = randomPredict(image_feature_array.T)
    y = y.reshape(row_num, column_num)
    # print(y.shape)
    # print(y)
    mask_image = np.zeros((img_RGB.shape[0], img_RGB.shape[1], 3), dtype=np.uint8)
    step = 2
    # width = width * step
    for i in range(0, row_num, step):
        for j in range(0, column_num, step):
            if(np.sum(y[i : i + step, j: j + step]) > 0.8 * step * step):
                 cv2.rectangle(mask_image, (i * width, j * width), (i * width + width * step, j * width + width * step), (0,0,255), -1)
            elif (np.sum(y[i : i + step, j: j + step]) > 0.7 * step * step):
                cv2.rectangle(mask_image, (i * width, j * width), (i * width + width * step, j * width + width * step), (0,255,255), -1)
            else:
                continue
                # cv2.rectangle(mask_image, (i * width, j * width), (i * width + width * step, j * width + width * step), (255,255,255), -1)
    # for i in range(0, row_num, step):
    #     for j in range(0, column_num, step):
    #         if(np.sum(y[i * column_num + j:i * column_num + j + step]) > 0.9 * step):
    #             cv2.rectangle(mask_image, (i * width, j * width), (i * width + width, j * width + width), (0,0,255), -1)
    #         elif (np.sum(y[i * column_num + j:i * column_num + j + step]) > 0.75 * step):
    #             cv2.rectangle(mask_image, (i * width, j * width), (i * width + width, j * width + width), (0,255,255), -1)
    #         else:
    #             cv2.rectangle(mask_image, (i * width, j * width), (i * width + width, j * width + width), (255,255,255), -1)
    print(mask_image.shape)
    print(img_RGB.shape)
    cv2.imwrite("mask_image.jpg", mask_image)
    result = cv2.addWeighted(img_RGB, 0.7, mask_image, 0.3, 0)
    cv2.imwrite("result_detected.JPG", result)


    
    
        



    

    




detector()