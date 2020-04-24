# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
df = pd.read_csv ("train.truth.csv")
train_datagen = ImageDataGenerator (rescale = 1. / 255)
test_datagen = ImageDataGenerator (rescale = 1. / 255)

train_generator = train_datagen.flow_from_dataframe(dataframe=df, directory="C:/Users/Eduardo/deeper_test/train",
                                                    x_col="fn", y_col="label", class_mode="categorical", 
                                                    target_size=(64,64), batch_size=32)
test_generator = test_datagen.flow_from_directory('C:/Users/Eduardo/deeper_test/test',
                                                  target_size = (64,64), batch_size = 32,
                                                  class_mode='categorical')

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(64,64,3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

opt = optimizers.RMSprop(lr=0.0001, rho=0.9)
model.compile(optimizer = opt, loss="categorical_crossentropy", metrics=["accuracy"])
model.fit_generator(train_generator , epochs=30, validation_data=test_generator, steps_per_epoch= 2500, validation_steps=500)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
model.save(filepath)
# Score trained model.
scores = model.evaluate(train_generator, test_generator, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])







