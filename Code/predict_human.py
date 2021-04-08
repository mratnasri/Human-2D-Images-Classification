from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sklearn.datasets import load_files
import numpy as np

# load and prepare the image

# test_dir = '../../trial data/Test'
# test_dir = '../../fruits-360/Test'


def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(480, 640))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 480, 640, 3)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

# load an image and predict the class


def run_example():
    # load the image
    img = load_image('../../sample_img.jpg')
    # load model
    model = load_model('../Models/human_2D_model9.h5')

    # predict the class
    #prediction = model.predict_classes(img)
    pred_prob = model.predict(img)
    pred_class = np.argmax(pred_prob, axis=-1)
    # print(fruit)
    print('predicted class: {0} '.format(
        pred_class[0]))


# entry point, run the example
run_example()
