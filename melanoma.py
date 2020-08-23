#!/usr/bin/env python
#! -*- coding: utf-8 -*-
from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import pygame.mixer
import numpy as np
import picamera
from PIL import Image
from time import sleep
import efficientnet.keras
import time

photo_filename = 'data.jpg'

def shutter():
    photofile = open(photo_filename, 'wb')
    print(photofile)

    # pi camera 用のライブラリーを使用して、画像を取得
    with picamera.PiCamera() as camera:
        #camera.resolution = (640,480)
        camera.resolution = (300,400)
        camera.start_preview()
        sleep(1.000)
        camera.capture(photofile)
        

def cosine_similarity(x1, x2):
    """
    test_dataと学習済み商品のコサイン類似度を算出
    n_dimはベクトルの次元　1000~1500程度
    x1: 対象の商品のベクトル   shape(1, n_dim)
    x2: 学習済みの商品のベクトル(hold_vector) shape(5, n_dim)
    return: 5つの商品に対するコサイン類似度 shape(1,5)
    """
    
    if x1.ndim == 1:
        x1 = x1[np.newaxis]
    if x2.ndim == 1:
        x2 = x2[np.newaxis]
    x1_norm = np.linalg.norm(x1, axis=1)
    x2_norm = np.linalg.norm(x2, axis=1)
    cosine_sim = np.dot(x1, x2.T)/(x1_norm*x2_norm+1e-10)
    return cosine_sim     
        
        
#model building
#IMAGE_SIZE = [128,128]

model = tf.keras.Sequential([
    efn.EfficientNetB5(
        input_shape=(*IMAGE_SIZE, 3),
        #weights='imagenet',
        weights='imagenet',
        include_top=False
    ),
    L.GlobalAveragePooling2D(),
    L.Dense(1024, activation = 'relu'), 
    L.Dropout(0.3), 
    L.Dense(512, activation= 'relu'), 
    L.Dropout(0.2), 
    L.Dense(256, activation='relu'), 
    L.Dropout(0.2), 
    L.Dense(128, activation='relu'), 
    L.Dropout(0.1), 
    L.Dense(1, activation='sigmoid')
])
        

if __name__ == '__main__':
    # モデル+重みを読込み
    #self_model = load_model('MobileNet_auto_fine3_150_3.h5')
    model.load_weights('models/complete_data_efficient_weights.h5')#('models/eff3_model.h5')
    
    lang = input('which langage?\n English:press "e"\n Flench:press "f"\n Japanese:press "j"')
    

    while True:
        basket = []
        money_sum = 0
        if lang == "e":
            key = input('Press "Enter" to scan products')
        elif lang == "f":
            key=input('Appuyez sur "Enter" pour numeriser les produits')
        else:
            key = input('商品をスキャンする場合は「Enter」を押して下さい')
        while True:
            # 画像の取得
            shutter()

            sleep(1)

            # 画像をモデルの入力用に加工
            img = Image.open(photo_filename)
            #img = img.resize((224, 224))
            #img_array = img_to_array(img)
            #img_array = img_array.astype('float32')/255.0
            #img_array = img_array.reshape((1,224,224,3))                      
            
            # predict
            #img_pred = self_model.predict(img_array)

            #model.load_weights('complete_data_efficient_weights.h5')

            #img_path = (jpg_name + '.jpg')
            img = img_to_array(load_img(img, target_size=(128,128)))
            img_nad = img_to_array(img)/255
            img_nad = img_nad[None, ...]

            #label=['homura','kyoko','madoka','mami','sayaka']
            pred = model.predict(img_nad, batch_size=1, verbose=0)
            score = np.max(pred)
            pred_label = label[np.argmax(pred[0])]
            print('name:',pred_label)
            print('AUC:',score)
            print("Malignant" if score>Threshold else "Benign")


            

