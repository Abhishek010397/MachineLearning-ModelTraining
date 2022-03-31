import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop


train = ImageDataGenerator(rescale = 1/255)
validation = ImageDataGenerator(rescale= 1/255)


###CAR DATASET
train_dataset = train.flow_from_directory('./training/training_car',
                                          target_size = (200,200),
                                          batch_size= 5,
                                          class_mode= 'binary')

print(train_dataset.class_indices)
validation_dataset = validation.flow_from_directory('./validation/validation_car',
                                          target_size = (200,200),
                                          batch_size= 5,
                                          class_mode= 'binary')

print(validation_dataset.class_indices)
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
                      steps_per_epoch = 5,
                      epochs = 30,
                      validation_data= validation_dataset)

####TRUCK_BUS_DATASET
train_truck_bus_dataset = train.flow_from_directory('./training/training_truck_bus',
                                                target_size=(200,200),
                                                batch_size=5,
                                                class_mode = 'binary')

print(train_truck_bus_dataset.class_indices)

validation_truck_bus_dataset = validation.flow_from_directory('./validation/validation_truck_bus',
                                          target_size = (200,200),
                                          batch_size= 5,
                                          class_mode= 'binary')

print(validation_truck_bus_dataset.class_indices)

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
                      steps_per_epoch = 5,
                      epochs = 30,
                      validation_data= validation_truck_bus_dataset)


##BIKES DATASET
train_bikes_scooty_dataset = train.flow_from_directory('./training/training_bikes_scooty',
                                                target_size=(200,200),
                                                batch_size=5,
                                                class_mode = 'binary')

validation_bikes_scooty_dataset = validation.flow_from_directory('./validation/validation_bikes_scooty',
                                          target_size = (200,200),
                                          batch_size= 5,
                                          class_mode= 'binary')

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
                      steps_per_epoch = 5,
                      epochs = 30,
                      validation_data= validation_bikes_scooty_dataset)


# ##AUTORICKSHAW DATASET
train_autorickshaw_dataset = train.flow_from_directory('./training/training_autorickshaw',
                                                target_size=(200,200),
                                                batch_size=5,
                                                class_mode = 'binary')

validation_autorickshaw_dataset = validation.flow_from_directory('./validation/validation_autorickshaw',
                                          target_size = (200,200),
                                          batch_size= 5,
                                          class_mode= 'binary')

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
                      steps_per_epoch = 5,
                      epochs = 30,
                      validation_data= validation_autorickshaw_dataset)


dir_path='./testing/'

for i in os.listdir(dir_path):
    img = image.load_img(dir_path+'/'+i, target_size = (200,200))
    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    images = np.vstack([X])
    result = model.predict(images)
    # print('Car Result',result)
    if result == 0:
        print('Car')
    else:
        result1 = model_truck_bus.predict(images)
        # print("Truck Result",result1)
        if result1 == 1:
            print('Truck Or Bus')
        else:
            result2=model_bikes_scooty.predict(images)
            # print('Bike Result',result2)
            if result2 == 0:
                print("Bike Or Scooty")
            else:
                result3=model_autorickshaw.predict(images)
                # print("AutoRickshaw Result",result3)
                if result3 == 0:
                    print('Autorickshaw')
                else:
                    print("Not A Vehicle")
    plt.imshow(img)
    plt.show()