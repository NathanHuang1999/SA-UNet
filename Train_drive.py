import os

import cv2
from keras.callbacks import TensorBoard, ModelCheckpoint
# import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc.pilutil import *
data_location = ''

# set data path

# use original dataset
# training_images_loc = data_location + 'DRIVE/train/images/'
# training_label_loc = data_location + 'DRIVE/train/labels/'
# validate_images_loc = data_location + 'DRIVE/validate/images/'
# validate_label_loc = data_location + 'DRIVE/validate/labels/'

# use my augmentation dataset
training_images_loc = data_location + 'DRIVE/na_train/images/'
training_label_loc = data_location + 'DRIVE/na_train/labels/'
validate_images_loc = data_location + 'DRIVE/na_validate/images/'
validate_label_loc = data_location + 'DRIVE/na_validate/labels/'


train_files = os.listdir(training_images_loc)
train_data = []
train_label = []
validate_files = os.listdir(validate_images_loc)
validate_data = []
validate_label = []

desired_size = 592

print(f"Processing {len(train_files)} train samples...")
for i in train_files:
    im = imread(training_images_loc + i)
    # label = imread(training_label_loc + i.split('_')[0] + '_manual1.png',mode="L")  # use original dataset
    label = imread(training_label_loc + i, mode="L")  # use my augmentation dataset
    old_size = im.shape[:2]  # old_size is in (height, width) format
    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    color2 = [0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    new_label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                   value=color2)

    train_data.append(cv2.resize(new_im, (desired_size, desired_size)))

    temp = cv2.resize(new_label, (desired_size, desired_size))
    _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
    train_label.append(temp)

print(f"Processing {len(validate_files)} validate samples...")
for i in validate_files:
    im = imread(validate_images_loc + i)
    # label = imread(validate_label_loc + i.split('_')[0] + '_manual1.png',mode="L")  # use original dataset
    label = imread(validate_label_loc + i, mode="L")  # use my augmentation dataset
    old_size = im.shape[:2]  # old_size is in (height, width) format
    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    color2 = [0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    new_label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                   value=color2)

    validate_data.append(cv2.resize(new_im, (desired_size, desired_size)))

    temp = cv2.resize(new_label, (desired_size, desired_size))
    _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
    validate_label.append(temp)

train_data = np.array(train_data)
train_label = np.array(train_label)

validate_data = np.array(validate_data)
validate_label = np.array(validate_label)

x_train = train_data.astype('float32') / 255.
y_train = train_label.astype('float32') / 255.
x_train = np.reshape(x_train, (
len(x_train), desired_size, desired_size, 3))  # adapt this if using `channels_first` image data format
y_train = np.reshape(y_train, (len(y_train), desired_size, desired_size, 1))  # adapt this if using `channels_first` im

x_validate = validate_data.astype('float32') / 255.
y_validate = validate_label.astype('float32') / 255.
x_validate = np.reshape(x_validate, (
len(x_validate), desired_size, desired_size, 3))  # adapt this if using `channels_first` image data format
y_validate = np.reshape(y_validate,
                        (len(y_validate), desired_size, desired_size, 1))  # adapt this if using `channels_first` im


TensorBoard(log_dir='./autoencoder', histogram_freq=0,
            write_graph=True, write_images=True)

from SA_UNet import *
model=SA_UNet(input_size=(desired_size,desired_size,3),start_neurons=16,lr=1e-4,keep_prob=0.82,block_size=7)
model.summary()
weight="Model/DRIVE/SA_UNet_220331.h5"
restore=True

if restore and os.path.isfile(weight):
    print(f"Loading weights from {weight}...")
    model.load_weights(weight)

new_path = "Model/DRIVE/SA_UNet_220331_2.h5"
model_checkpoint = ModelCheckpoint(new_path, monitor='val_accuracy', verbose=1, save_best_only=False)


history=model.fit(x_train, y_train,
                epochs=50, #first 100 with lr=1e-3,,and last 50 with lr=1e-4
                batch_size=3,
                # validation_split=0.05,
                validation_data=(x_validate, y_validate),
                shuffle=True,
                callbacks= [TensorBoard(log_dir='./autoencoder'), model_checkpoint],)

print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('SA-UNet Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='lower right')
plt.show()


