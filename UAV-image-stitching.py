                           # -*- coding: utf-8 -*-
#Import library
#import libraries
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils




def maximum_internal_rectangle():
    img = cv2.imread("result_RGB.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    ret, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY) 
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    # contour = contours[0].reshape(len(contours[0]), 2) 
    max_rect = (0,0,0,0)
    max_area = 0
    for i in range(0, len(contours), 10):
        x1, y1 = contours[i][0][0]
        for j in range(i, len(contours),10):
            x2, y2 = contours[j][0][0]            
            area = abs(y2 - y1) * abs(x2 - x1)
            if (area > max_area) and (np.sum(img_bin[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)]) > 0.98 * area * 255):
                max_area = area
                max_rect = (x1, y1, x2, y2)
    x1, y1, x2, y2 = max_rect
    cropped_image = img[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)]
    cv2.imwrite("result_RGB_FINAL.jpg", cropped_image)

    img = cv2.imread("result_R.TIF")
    cropped_image = img[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)]
    cropped_image = imutils.resize(cropped_image, width=3000)
    cv2.imwrite("result_R_FINAL.TIF", cropped_image)

    img = cv2.imread("result_G.TIF")
    cropped_image = img[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)]
    cropped_image = imutils.resize(cropped_image, width=3000)
    cv2.imwrite("result_G_FINAL.TIF", cropped_image)

    img = cv2.imread("result_RE.TIF")
    cropped_image = img[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)]
    cropped_image = imutils.resize(cropped_image, width=3000)
    cv2.imwrite("result_RE_FINAL.TIF", cropped_image)

    img = cv2.imread("result_NIR.TIF")
    cropped_image = img[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)]
    cropped_image = imutils.resize(cropped_image, width=3000)
    cv2.imwrite("result_NIR_FINAL.TIF", cropped_image)

def warpImages(img1, img2, H):
  rows1, cols1 = img1.shape[:2]
  rows2, cols2 = img2.shape[:2]

  list_of_points_1 = np.float32([[0,0], [0, rows1],[cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2) #coordinates of a reference image
  temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2) #coordinates of second image

  # When we have established a homography we need to warp perspective
  # Change field of view
  list_of_points_2 = cv2.perspectiveTransform(temp_points, H)#calculate the transformation matrix

  list_of_points = np.concatenate((list_of_points_1,list_of_points_2), axis=0)

  [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
  [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
  
  translation_dist = [-x_min,-y_min]
  
  H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

  output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
  output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1

  return output_img







import glob

path = sorted(glob.glob("*D.JPG"))
path_R = sorted(glob.glob("*R.TIF"))
path_G = sorted(glob.glob("*G.TIF"))
path_RE = sorted(glob.glob("*RE.TIF"))
path_NIR = sorted(glob.glob("*NIR.TIF"))





img_list = []
img_list_R = []
img_list_G = []
img_list_NIR = []
img_list_RE = []
for img in path:
    n = cv2.imread(img)
    n = imutils.resize(n, width=3000)
    img_list.append(n)
   
for img_R in path_R:
    n_R = cv2.imread(img_R)
    n_R = imutils.resize(n_R, width=3000)
    img_list_R.append(n_R)

for img_G in path_G:
    n_G = cv2.imread(img_G)
    n_G = imutils.resize(n_G, width=3000)
    img_list_G.append(n_G)

for img_RE in path_RE:
    n_RE = cv2.imread(img_RE)
    n_RE = imutils.resize(n_RE, width=3000)
    img_list_RE.append(n_RE)

for img_NIR in path_NIR:
    n_NIR = cv2.imread(img_NIR)
    n_NIR = imutils.resize(n_NIR, width=3000)
    img_list_NIR.append(n_NIR)

    


    
    
"""Functions for stitching"""

#Use ORB detector to extract keypoints
orb = cv2.ORB_create(nfeatures=2000)
while True:
  img1=img_list.pop(0)
  img2=img_list.pop(0)
  img1_R  = img_list_R.pop(0)
  img2_R = img_list_R.pop(0)
  img1_G  = img_list_G.pop(0)
  img2_G = img_list_G.pop(0)
  img1_RE  = img_list_RE.pop(0)
  img2_RE = img_list_RE.pop(0)
  img1_NIR  = img_list_NIR.pop(0)
  img2_NIR = img_list_NIR.pop(0)
# Find the key points and descriptors with ORB
  keypoints1, descriptors1 = orb.detectAndCompute(img1, None)#descriptors are arrays of numbers that define the keypoints
  keypoints2, descriptors2 = orb.detectAndCompute(img2, None)


# Create a BFMatcher object to match descriptors
# It will find all of the matching keypoints on two images
  bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)#NORM_HAMMING specifies the distance as a measurement of similarity between two descriptors

# Find matching points
  matches = bf.knnMatch(descriptors1, descriptors2,k=2)

  all_matches = []
  for m, n in matches:
    all_matches.append(m)
# Finding the best matches
  good = []
  for m, n in matches:
    if m.distance < 0.6 * n.distance:#Threshold
        good.append(m)

# Set minimum match condition
  MIN_MATCH_COUNT = 5

  if len(good) > MIN_MATCH_COUNT:
    
    # Convert keypoints to an argument for findHomography
    src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    # Establish a homography
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    
    result = warpImages(img2, img1, M)
    # img1_R = img_list_R_index[img_list_index.index(img1)]
    # img2_R = img_list_R_index[img_list_index.index(img2)]
    result_R = warpImages(img2_R, img1_R, M)
    result_G = warpImages(img2_G, img1_G, M)
    result_RE = warpImages(img2_RE, img1_RE, M)
    result_NIR = warpImages(img2_NIR, img1_NIR, M)
    
    img_list.insert(0,result)
    img_list_R.insert(0,result_R)
    img_list_G.insert(0,result_G)
    img_list_RE.insert(0,result_RE)
    img_list_NIR.insert(0,result_NIR)
    
    if len(img_list)==1:
      break
cv2.imwrite("result_RGB.jpg", result)
cv2.imwrite("result_R.TIF", result_R)
cv2.imwrite("result_G.TIF", result_G)
cv2.imwrite("result_RE.TIF", result_RE)
cv2.imwrite("result_NIR.TIF", result_NIR)
result2 = cv2.cvtColor(result, cv2.COLOR_BGR2RGB )  
# cv2.imwrite("result_RGB2.jpg", result2)
maximum_internal_rectangle()

# maximum_internal_rectangle("result_RGB.jpg")
# plt.imshow(result)
# plt.show()

