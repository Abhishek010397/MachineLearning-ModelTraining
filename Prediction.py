import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing import image

car_model=tf.keras.models.load_model('./Model/car_model.h5')
truck_bus_model=tf.keras.models.load_model('./Model/truck_bus_model.h5')
bikes_scooty_model=tf.keras.models.load_model('./Model/bikes_scooty_model.h5')
autorickshaw_model=tf.keras.models.load_model('./Model/autorickshaw_model.h5')

dir_path='./Model/testing/'

for i in os.listdir(dir_path):
    img = image.load_img(dir_path+'/'+i, target_size = (200,200))
    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    images = np.vstack([X])
    result = car_model.predict(images)
    # print('Car Result',result)
    if result == 0:
        print('Car')
    else:
        result1 = truck_bus_model.predict(images)
        # print("Truck Result",result1)
        if result1 == 1:
            print('Truck Or Bus')
        else:
            result2=bikes_scooty_model.predict(images)
            # print('Bike Result',result2)
            if result2 == 0:
                print("Bike Or Scooty")
            else:
                result3=autorickshaw_model.predict(images)
                # print("AutoRickshaw Result",result3)
                if result3 == 0:
                    print('Autorickshaw')
                else:
                    print("Not A Vehicle")
    plt.imshow(img)
    plt.show()