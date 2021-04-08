import numpy as np
from numpy import mean, std
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import sys
import cv2
from glob import glob
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import skimage
import skimage.transform
import random

sess=tf.Session()
all_images = '../../Complete 2D Human Images Dataset/*.jpg'
#classes = [x for x in range(200)]


def load_dataset(path):
    img_names = glob(all_images)
    images = []
    names = []
    for fn in img_names:
        img = load_img(fn)
        # print(img.shape)
        img = img_to_array(img)
        images.append(img)
        name = fn.split("\\")[-1]
        names.append(name)
    return images, names


images, names = load_dataset(all_images)

# images=np.array(images)
"""img = load_img('data/train/cats/cat.0.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)"""


def crop_img(img):
    original_size = list(img.shape)
    x = img
    # Use the following commands to perform random crops
    crop_size = [int(0.8*original_size[0]), int(0.5*original_size[1]), original_size[2]]
    seed = np.random.randint(1234)
    x = tf.random_crop(x, size=crop_size, seed=seed)
    output = tf.image.resize_images(x, size=[original_size[0],original_size[1]])
    tf.cast(output, dtype=tf.uint8)
    return output

def rotate_img(img):
    #rot = skimage.transform.rotate(img, angle = random.randint(1,5), mode='constant', cval=1)
    rot = skimage.transform.rotate(img, angle = random.randint(1,5), mode='edges')
    return rot


sess.run(tf.initialize_all_variables())

for j in range(len(images)):
    cropped = crop_img(images[j])
    cropped = 255*cropped/tf.math.reduce_max(cropped)
    cropped=tf.cast(cropped, dtype=tf.uint8)
    cropped_img=tf.image.encode_png(cropped)
    crop_write=tf.write_file('../../dataAugmentation_ver2/'+names[j].replace('.jpg', '_cropped.png'),cropped_img)
    rotated = rotate_img(images[j])
    rotated=tf.cast(rotated,dtype= tf.uint8)
    rotated=tf.image.encode_png(rotated)
    writer = tf.write_file('../../dataAugmentation_ver2/'+names[j].replace('.jpg', '_rotated.png'),rotated)
    sess.run(crop_write)
    sess.run(writer)
