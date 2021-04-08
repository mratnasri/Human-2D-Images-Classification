import numpy as np
from numpy import mean, std
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import sys
import cv2
from glob import glob
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
datagen = ImageDataGenerator(
    # rotation_range=5,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    zoom_range=0.2,
    # horizontal_flip=False,
    # fill_mode=None,
    rescale=1.10,
    dtype='uint8'
)

all_images = '../../Complete 2D Human Images Dataset/*.jpg'
categories_n = 200
#classes = [x for x in range(200)]


def load_dataset(path):
    img_names = glob(all_images)
    images = []
    targets = []
    for fn in img_names:
        img = load_img(fn)
        # print(img.shape)
        img = img_to_array(img)
        images.append(img)
        target = fn.split("\\")[-1].split("-")[0]
        targets.append(target)
    return images, targets


images, targets = load_dataset(all_images)
# images=np.array(images)
"""img = load_img('data/train/cats/cat.0.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)"""

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
for j in range(len(images)):
    # images[j]=np.array(images[j])
    images[j] = images[j].reshape((1,)+images[j].shape)
    # print(images[j].shape)
    i = 0
    for batch in datagen.flow(images[j], batch_size=1,
                              save_to_dir='../../dataAugmentation', save_prefix=targets[j], save_format='jpg'):
        # plt.figure(i)
        #imgplot = plt.imshow(array_to_img(batch[0]))
        # batch.astype(np.uint8)
        i += 1
        if i >= 2:
            break  # otherwise the generator would loop indefinitely)
