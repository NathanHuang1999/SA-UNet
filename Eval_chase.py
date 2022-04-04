import os
import cv2
import numpy as np
from scipy.misc.pilutil import *
from PIL import Image
from SA_UNet import *
import time
import skimage.io as io

# Part 1:
# read test images and labels, then preprocess them
data_location = ''

testing_images_loc = data_location + "./CHASE_all_test/images/"
testing_labels_loc = data_location + "./CHASE_all_test/labels/"

test_image_files = os.listdir(testing_images_loc)
test_images_data = []
test_labels_data = []

desired_size = 1008

for image in test_image_files:

    im = imread(testing_images_loc + image)
    label = imread(testing_labels_loc + image.split('.')[0] + "_1stHO.png", mode="L")

    # padding
    old_size = im.shape[:2]  # old_size is in (height, width) format
    delta_w = desired_size - old_size[1]  # 1008-999=9
    delta_h = desired_size - old_size[0]  # 1008-960=48

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)  # top=24, bottom=24
    left, right = delta_w // 2, delta_w - (delta_w // 2)  # left=4, right=5

    color = [0, 0, 0]
    color2 = [0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    new_label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                   value=color2)

    test_images_data.append(cv2.resize(new_im, (desired_size, desired_size)))

    temp = cv2.resize(new_label,
                      (desired_size, desired_size))
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
weight="Model/DRIVE/SA_UNet_220331.h5"
model.load_weights(weight)


# Part 3:
# RUN!

threshold_of_prediction = 0.5  # use this value to make bi-classification

# model.evaluate()
prediction_base = "./DRIVE/prediction/"
if not os.path.exists(prediction_base):
    os.mkdir(prediction_base)

prediction_loc = prediction_base + "CHASE_{exp_date}/".format(exp_date=time.strftime('%y%m%d_%H%M%S', time.localtime(time.time())))
if not os.path.exists(prediction_loc):
    os.mkdir(prediction_loc)

pred = model.predict(x_test, batch_size=3)  # np.array, shape:(20, 1008, 1008, 1)
# the following code should be modified to output images of the original size
for result_idx in range(28):

    # produce full size images
    # remove the padding pixels around the image
    result_pred = pred[result_idx][24:desired_size - 24, 4:desired_size - 5]  # shape:(960,999)

    # save probability pictures
    io.imsave(prediction_loc + "prob_{idx}.png".format(idx=result_idx + 1), result_pred)

    # save binary pictures
    # create a array for containing the image
    result_img = np.zeros((960 * 999, 3), np.uint8)
    result_img[result_pred.ravel() > threshold_of_prediction] = [255, 255, 255]
    result_img.shape = (960, 999, 3)

    Image.fromarray(result_img).save(
        prediction_loc + "{idx}_pred_{th}.png".format(idx=result_idx + 1, th=threshold_of_prediction))

print("Prediction is done!")



