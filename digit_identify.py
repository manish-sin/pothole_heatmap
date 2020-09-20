from PIL import Image
import os,re, pandas as pd
import matplotlib.pyplot as plt


import cv2
import numpy as np

def get_image(img_no):
  img_loc = "frames/%s"%(img_no)
  main_img = cv2.imread(img_loc)
  cv2.imshow("main image", main_img)
  image = Image.open(img_loc)
  # image.show()
  #crop_lat(image, img_no)
  crop_long(image, img_no)

def crop_long(image, img_no):
  cropped_image_long1 = image.crop((405, 1132, 430, 1166))
  cropped_image_long2 = image.crop((432, 1132, 457, 1166))
  cropped_image_long3 = image.crop((470, 1132, 495, 1166))
  cropped_image_long4 = image.crop((498, 1132, 523, 1166))
  cropped_image_long5 = image.crop((525, 1132, 550, 1166))
  cropped_image_long6 = image.crop((551, 1132, 576, 1166))
  cropped_image_long7 = image.crop((578, 1132, 603, 1166))
  cropped_image_long8 = image.crop((604, 1132, 629, 1166))
  cropped_img_loc_long1 = "coordinates/all_digits/long_crop_1_%s"%(img_no)
  cropped_img_loc_long2 = "coordinates/all_digits/long_crop_2_%s"%(img_no)
  cropped_img_loc_long3 = "coordinates/all_digits/long_crop_3_%s"%(img_no)
  cropped_img_loc_long4 = "coordinates/all_digits/long_crop_4_%s"%(img_no)
  cropped_img_loc_long5 = "coordinates/all_digits/long_crop_5_%s"%(img_no)
  cropped_img_loc_long6 = "coordinates/all_digits/long_crop_6_%s"%(img_no)
  cropped_img_loc_long7 = "coordinates/all_digits/long_crop_7_%s"%(img_no)
  cropped_img_loc_long8 = "coordinates/all_digits/long_crop_8_%s"%(img_no)
  cropped_image_long1.save(cropped_img_loc_long1)
  cropped_image_long2.save(cropped_img_loc_long2)
  cropped_image_long3.save(cropped_img_loc_long3)
  cropped_image_long4.save(cropped_img_loc_long4)
  cropped_image_long5.save(cropped_img_loc_long5)
  cropped_image_long6.save(cropped_img_loc_long6)
  cropped_image_long7.save(cropped_img_loc_long7)
  cropped_image_long8.save(cropped_img_loc_long8)
  long_img_crop1 = cv2.imread(cropped_img_loc_long1)
  long_img_crop1  = cv2.copyMakeBorder(long_img_crop1 , 10, 10, 10, 10, cv2.BORDER_CONSTANT)
  long_img_crop2 = cv2.imread(cropped_img_loc_long2)
  long_img_crop2 = cv2.copyMakeBorder(long_img_crop2, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
  long_img_crop3 = cv2.imread(cropped_img_loc_long3)
  long_img_crop3 = cv2.copyMakeBorder(long_img_crop3, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
  long_img_crop4 = cv2.imread(cropped_img_loc_long4)
  long_img_crop4 = cv2.copyMakeBorder(long_img_crop4, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
  long_img_crop5 = cv2.imread(cropped_img_loc_long5)
  long_img_crop5 = cv2.copyMakeBorder(long_img_crop5, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
  long_img_crop6 = cv2.imread(cropped_img_loc_long6)
  long_img_crop6 = cv2.copyMakeBorder(long_img_crop6, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
  long_img_crop7 = cv2.imread(cropped_img_loc_long7)
  long_img_crop7 = cv2.copyMakeBorder(long_img_crop7, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
  long_img_crop8 = cv2.imread(cropped_img_loc_long8)
  long_img_crop8 = cv2.copyMakeBorder(long_img_crop8, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
  img_concate_Hori = np.concatenate((long_img_crop1,long_img_crop2,long_img_crop3,long_img_crop4,long_img_crop5,long_img_crop6,long_img_crop7,long_img_crop8), axis=1)
  cv2.imshow('coordinates',img_concate_Hori)


get_image("frame0.jpg")
cv2.waitKey(0)
cv2.destroyAllWindows()
