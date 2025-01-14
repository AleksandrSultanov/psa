# -*- coding: utf-8 -*-
"""Make_dataset_SOBEL.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zDmdfh-GIL9jmszFbo0QnNB4bjVWJhUA
"""

import pandas as pd
import cv2 as cv
import os
from PIL import Image
import numpy as np
from PIL import ImageFilter
from google.colab import drive
from google.colab.patches import cv2_imshow
from matplotlib import pyplot as plt
drive.mount('/content/drive')
os.chdir('/content/drive/My Drive/Dataset')


#Загружаем данные
fold0 = pd.read_csv("fold_0_data.txt",sep = "\t" )
fold1 = pd.read_csv("fold_1_data.txt",sep = "\t")
fold2 = pd.read_csv("fold_2_data.txt",sep = "\t")
fold3 = pd.read_csv("fold_3_data.txt",sep = "\t")
fold4 = pd.read_csv("fold_4_data.txt",sep = "\t")


#Добавьте данные из всех этих файлов в один массив данных pandas и распечатайте информацию о ней.
total_data = pd.concat([fold1], ignore_index=True)
print(total_data.shape)
total_data.info()

#Цикл по всем изображениям
for row in total_data.iterrows():
    #Загружаем исходное изображение
    image = cv.imread("Новая папка/"+row[1].user_id+"/landmark_aligned_face."+str(row[1].face_id)+"."+row[1].original_image, cv.IMREAD_UNCHANGED)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    output = cv.Sobel(image,cv.CV_8U,1,0,ksize=5)

    #Сохраняем новое изображение
    cv.imwrite("Новая папка/"+row[1].user_id+"/landmark_aligned_face."+str(row[1].face_id)+".SOBEL_"+row[1].original_image, output)

    #Вносим информацию об изображении в CSV файл
    frame = pd.DataFrame([[row[1].user_id, "SOBEL_"+row[1].original_image, row[1].face_id, row[1].age, row[1].gender, row[1].x, row[1].y,
                           row[1].dx, row[1].dy, row[1].tilt_ang, row[1].fiducial_yaw_angle, row[1].fiducial_score]])

    frame.to_csv('fold_1_data.txt', mode='a', header=False, sep = "\t", index=False)

print("GOOD!")