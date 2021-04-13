import os
import os.path
import random
from sklearn.datasets import load_files
import numpy as np
import keras
from keras.utils import to_categorical
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from numpy import mean, std
#from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, top_k_accuracy_score, ConfusionMatrixDisplay
import sys
import cv2
from glob import glob

# 480x640 images
# 131 classes

#train_dir = '../../trial data/Training'
all_images = '../../Complete 2D Human Images Dataset/*.jpg'
dataAugmentation = '../../dataAugmentation_ver3/*.png'
noise_augmentation = '../../dataAugmentation_noise/*.png'
translate_flip_augmentation = '../../dataAugmentation_translate_flip/*.png'
categories_n = 200
#classes = [x for x in range(200)]

images = []
targets = []


def load_dataset(path, images, targets):
    img_names = glob(path)
    for fn in img_names:
        img = cv2.imread(fn)
        #img = load_img(fn)
        # print(img.shape)
        images.append(img)
        target = fn.split("/")[-1].split("-")[0]  # \\ for windows, / for linux
        targets.append(target)
    return images, targets


images, targets = load_dataset(all_images, images, targets)
images, targets = load_dataset(dataAugmentation, images, targets)
images, targets = load_dataset(translate_flip_augmentation, images, targets)
noise_images = []
noise_targets = []
noise_images, noise_targets = load_dataset(
    noise_augmentation, noise_images, noise_targets)
images_num = len(images)
targets = [int(ele) - 1 for ele in targets]
noise_targets = [int(ele) - 1 for ele in noise_targets]
print("Number of total samples = ", images_num)
print("Number of Noisy images = ", len(noise_images))
# print(targets)

# convert to grayscale


def convert_gray(images):
    gray_images = []
    for img in images:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #gray_img = cv2.equalizeHist(gray_img)
        gray_images.append(gray_img)
    return gray_images


gray_images = convert_gray(images)
noise_gray_images = convert_gray(noise_images)

#cv2.imwrite("../../stretched.png", gray_images[2])
# split into training, testing and validation
x_train, x_test, y_train, y_test = train_test_split(
    gray_images, targets, test_size=0.3, shuffle=True, stratify=targets)
x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, test_size=0.5, shuffle=True, stratify=y_test)

x_train.extend(noise_gray_images)
y_train.extend(noise_targets)
temp = list(zip(x_train, y_train))
random.shuffle(temp)
x_train, y_train = zip(*temp)

y_train_ohe = to_categorical(y_train, categories_n)
y_val_ohe = to_categorical(y_val, categories_n)


# preprocessing

#x_train = np.array(convert_img_to_array(x_train))
x_train = np.array(x_train)
print('Training set shape : ', x_train.shape)
#x_val = np.array(convert_img_to_array(x_val))
x_val = np.array(x_val)
print('Validation set shape : ', x_val.shape)
#x_test = np.array(convert_img_to_array(x_test))
x_test = np.array(x_test)
print('Test set shape : ', x_test.shape)

x_train = x_train.reshape((x_train.shape[0], 480, 640, 1))
x_val = x_val.reshape((x_val.shape[0], 480, 640, 1))
x_test = x_test.reshape((x_test.shape[0], 480, 640, 1))

for img in x_train:
    img = 255*(img-np.min(img))/(np.max(img)-np.min(img))
    img = img.astype(np.uint8)

for img in x_val:
    img = 255*(img-np.min(img))/(np.max(img)-np.min(img))
    img = img.astype(np.uint8)

for img in x_test:
    img = 255*(img-np.min(img))/(np.max(img)-np.min(img))
    img = img.astype(np.uint8)

#cv2.imwrite("../../stretched2.jpg", x_train[2])
# normalization
x_train = x_train.astype('float32')
x_train = x_train/255
x_val = x_val.astype('float32')
x_val = x_val/255
x_test = x_test.astype('float32')
x_test = x_test/255


def model_config():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform', input_shape=(480, 640, 1)))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(300, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(200, activation='softmax'))
    # compile the model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy()])
    return model

# evaluate model using 5-fold cross validation


def train_model(datax, datay, valx, valy):
    model = model_config()
    # fit model
    history = model.fit(datax, datay, epochs=30, batch_size=32,
                        validation_data=(valx, valy), verbose=2)
    print('Accuracy: mean=%.3f std=%.3f, n=%d' %
          (mean(history.history['accuracy'])*100, std(history.history['accuracy'])*100, len(history.history['accuracy'])))
    print('Top-5 Accuracy: mean=%.3f std=%.3f, n=%d' %
          (mean(history.history['top_k_categorical_accuracy'])*100, std(history.history['top_k_categorical_accuracy'])*100, len(history.history['top_k_categorical_accuracy'])))
    print('Validation Accuracy: mean=%.3f std=%.3f, n=%d' %
          (mean(history.history['val_accuracy'])*100, std(history.history['val_accuracy'])*100, len(history.history['val_accuracy'])))
    print('Validation Top-5 Accuracy: mean=%.3f std=%.3f, n=%d' %
          (mean(history.history['val_top_k_categorical_accuracy'])*100, std(history.history['val_top_k_categorical_accuracy'])*100, len(history.history['val_top_k_categorical_accuracy'])))

    return history, model


history, model = train_model(x_train, y_train_ohe, x_val, y_val_ohe)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

"""plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Categorical Crossentropy')
plt.ylim([0, max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
# plt.show()
plt.savefig('../../Outputs/model23_graph.png')"""

model.save('../Models/human_2D_model23.h5')
print("saved")

# evaluation
print("Model evaluation on test dataset: ")
pred_prob = model.predict(x_test)
#pred_class = model.predict_classes(x_test)
pred_class = np.argmax(pred_prob, axis=-1)

# metrics
accuracy = accuracy_score(y_test, pred_class)
k_accuracy = top_k_accuracy_score(y_test, pred_prob, k=5)
print('accuracy =  %.3f' % (accuracy * 100.0),
      'top-5 accuracy = %.3f' % (k_accuracy*100))
"""precision = precision_score(y_test, pred_class)
recall = recall_score(y_test, pred_class)"""
report = classification_report(y_test, pred_class)
print("Classification Report: ")
print(report)
"""f1 = f1_score(y_test, pred_class,average='macro')
print("f1 score: ", f1)"""
confusionMatrix = confusion_matrix(
    y_test, pred_class)  # row(true), column(predicted)
np.set_printoptions(threshold=sys.maxsize)
print("Confusion matrix: ")
print(confusionMatrix)
np.set_printoptions(threshold=False)
#cm_labels = [x for x in range(20)]
"""disp = ConfusionMatrixDisplay(
    confusion_matrix=confusionMatrix)
disp.plot()

# plt.savefig('../../Outputs/model15_confusionMatrix.png')
plt.show()"""
print("end")
