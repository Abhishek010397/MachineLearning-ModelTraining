import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop


train = ImageDataGenerator(rescale = 1/255)



###CAR DATASET
train_dataset = train.flow_from_directory('./training/training_car',
                                          target_size = (200,200),
                                          batch_size= 5,
                                          class_mode= 'binary')



model = tf.keras.models.Sequential([ tf.keras.layers.Conv2D(16,(3,3),activation = 'relu', input_shape=(200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                     #
                                    tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                     #
                                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2, 2),
                                     #
                                    tf.keras.layers.Flatten(),
                                     ##
                                    tf.keras.layers.Dense(512, activation  = 'relu'),
                                     ##
                                    tf.keras.layers.Dense(1,activation='sigmoid')
                                    ])

model.compile(loss='binary_crossentropy',
              optimizer= RMSprop(learning_rate=0.001),
              metrics = ['accuracy'])

model_fit = model.fit(train_dataset,
                      steps_per_epoch = 15,
                      epochs = 40,
                      validation_data= None)



####TRUCK_BUS_DATASET
train_truck_bus_dataset = train.flow_from_directory('./training/training_truck_bus',
                                                target_size=(200,200),
                                                batch_size=5,
                                                class_mode = 'binary')

model_truck_bus = tf.keras.models.Sequential([ tf.keras.layers.Conv2D(16,(3,3),activation = 'relu', input_shape=(200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                     #
                                    tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                     #
                                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2, 2),
                                     #
                                    tf.keras.layers.Flatten(),
                                     ##
                                    tf.keras.layers.Dense(512, activation  = 'relu'),
                                     ##
                                    tf.keras.layers.Dense(1,activation='sigmoid')
                                    ])

model_truck_bus.compile(loss='binary_crossentropy',
              optimizer= RMSprop(learning_rate=0.001),
              metrics = ['accuracy'])

model_fit_truck_bus = model_truck_bus.fit(train_truck_bus_dataset,
                      steps_per_epoch = 15,
                      epochs = 40,
                      validation_data= None)


##BIKES DATASET
train_bikes_scooty_dataset = train.flow_from_directory('./training/training_bikes_scooty',
                                                target_size=(200,200),
                                                batch_size=5,
                                                class_mode = 'binary')

model_bikes_scooty = tf.keras.models.Sequential([ tf.keras.layers.Conv2D(16,(3,3),activation = 'relu', input_shape=(200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                     #
                                    tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                     #
                                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2, 2),
                                     #
                                    tf.keras.layers.Flatten(),
                                     ##
                                    tf.keras.layers.Dense(512, activation  = 'relu'),
                                     ##
                                    tf.keras.layers.Dense(1,activation='sigmoid')
                                    ])

model_bikes_scooty.compile(loss='binary_crossentropy',
              optimizer= RMSprop(learning_rate=0.001),
              metrics = ['accuracy'])

model_fit_bikes_scooty = model_bikes_scooty.fit(train_bikes_scooty_dataset,
                      steps_per_epoch = 15,
                      epochs = 40,
                      validation_data= None)


# ##AUTORICKSHAW DATASET
train_autorickshaw_dataset = train.flow_from_directory('./training/training_autorickshaw',
                                                target_size=(200,200),
                                                batch_size=5,
                                                class_mode = 'binary')

model_autorickshaw = tf.keras.models.Sequential([ tf.keras.layers.Conv2D(16,(3,3),activation = 'relu', input_shape=(200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                     #
                                    tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                     #
                                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2, 2),
                                     #
                                    tf.keras.layers.Flatten(),
                                     ##
                                    tf.keras.layers.Dense(512, activation  = 'relu'),
                                     ##
                                    tf.keras.layers.Dense(1,activation='sigmoid')
                                    ])

model_autorickshaw.compile(loss='binary_crossentropy',
              optimizer= RMSprop(learning_rate=0.001),
              metrics = ['accuracy'])

model_fit_autorickshaw = model_autorickshaw.fit(train_autorickshaw_dataset,
                      steps_per_epoch = 15,
                      epochs = 40,
                      validation_data= None)

###MODEL SAVING
car_h5_model= './car_model.h5'
truck_bus_h5_model= './truck_bus_model.h5'
bikes_scooty_h5_model= './bikes_scooty_model.h5'
autorickshaw_h5_model= './autorickshaw_model.h5'
tf.keras.models.save_model(model, car_h5_model)
tf.keras.models.save_model(model_truck_bus, truck_bus_h5_model)
tf.keras.models.save_model(model_bikes_scooty, bikes_scooty_h5_model)
tf.keras.models.save_model(model_autorickshaw, autorickshaw_h5_model)