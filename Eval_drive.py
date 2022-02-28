import os
import cv2
import numpy as np
from scipy.misc.pilutil import *
import matplotlib.pyplot as plt
from SA_UNet import *
import time

# Part 1:
# read test images and labels, then preprocess them
data_location = ""

testing_images_loc = data_location + "DRIVE/test/images/"
testing_labels_loc = data_location + "DRIVE/test/labels/"

test_image_files = os.listdir(testing_images_loc)
test_images_data = []
test_labels_data = []

desired_size = 592

for image in test_image_files:

    im = imread(testing_images_loc + image)
    label = imread(testing_labels_loc + image.split('_')[0] + '_manual1.png', mode="L")

    # padding
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

    test_images_data.append(cv2.resize(new_im, (desired_size, desired_size)))
    temp = cv2.resize(new_label, (desired_size, desired_size))
    _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
    test_labels_data.append(temp)

test_images_data = np.array(test_images_data)
test_labels_data = np.array(test_labels_data)

x_test = test_images_data.astype('float32') / 255.
y_test = test_labels_data.astype('float32') / 255.
x_test = np.reshape(x_test, (
    len(x_test), desired_size, desired_size, 3))  # adapt this if using `channels_first` image data format
y_test = np.reshape(y_test, (len(y_test), desired_size, desired_size, 1))  # adapt this if using `channels_first` im


# Part 2:
# load model and parameters

model=SA_UNet(input_size=(desired_size,desired_size,3),start_neurons=16,lr=1e-3,keep_prob=0.82,block_size=7)
weight="Model/DRIVE/SA_UNet.h5"
model.load_weights(weight)


# Part 3:
# RUN!

# model.evaluate()
prediction_base = "./DRIVE/prediction/"
if not os.path.exists(prediction_base):
    os.mkdir(prediction_base)

prediction_loc = prediction_base + "{exp_date}/".format(exp_date=time.strftime('%y%m%d_%H%M%S', time.localtime(time.time())))
if not os.path.exists(prediction_loc):
    os.mkdir(prediction_loc)

pred = model.predict(x_test, batch_size=3)  # np.array, shape:(20, 592, 592, 1)
for result_idx in range(20):
    plt.imshow(pred[result_idx].squeeze(2), cmap="gray")
    plt.savefig(prediction_loc + "{idx}_pred.png".format(idx=result_idx+1))

print("Prediction is done!")



