from glob import glob
from keras.preprocessing.image import img_to_array, load_img
import tensorflow.compat.v1 as tf
import tensorflow_addons as tfa
tf.disable_v2_behavior()

sess = tf.Session()
all_images = '../../Complete 2D Human Images Dataset/*.jpg'


def load_dataset(path):
    img_names = glob(all_images)
    images = []
    names = []
    for fn in img_names:
        img = load_img(fn)
        # print(img.shape)
        img = img_to_array(img)
        images.append(img)
        name = fn.split("/")[-1]  # \\ for windows, / for unix
        names.append(name)
    return images, names


images, names = load_dataset(all_images)


def translate_img(img):
    original_size = list(img.shape)
    x = img
    w = tf.random.uniform(shape=[],
                          minval=-original_size[0]*0.3, maxval=original_size[0]*0.3, dtype=tf.float32)
    h = tf.random.uniform(shape=[],
                          minval=1, maxval=original_size[1]*0.2, dtype=tf.float32)
    translated = tfa.image.translate(
        x, [w, h], interpolation='nearest', fill_mode='nearest')
    #translated = tf.cast(translated, dtype=tf.uint8)
    return translated


sess.run(tf.initialize_all_variables())

for j in range(len(images)):
    translated = translate_img(images[j])
    translated = 255*(translated-tf.math.reduce_min(translated)) / \
        (tf.math.reduce_max(translated)-tf.math.reduce_min(translated))
    translated = tf.cast(translated, dtype=tf.uint8)
    translated_img = tf.image.encode_png(translated)
    path = '../../dataAugmentation_translate_flip/' + \
        names[j].replace('.jpg', '_translate.png')
    # print(path)
    translate_write = tf.write_file(path, translated_img)
    # print("done")
    sess.run(translate_write)
    flipped = tf.image.flip_left_right(images[j])
    flipped = 255*(flipped-tf.math.reduce_min(flipped)) / \
        (tf.math.reduce_max(flipped)-tf.math.reduce_min(flipped))
    flipped = tf.cast(flipped, dtype=tf.uint8)
    flipped_img = tf.image.encode_png(flipped)
    path = '../../dataAugmentation_translate_flip/' + \
        names[j].replace('.jpg', '_flipped.png')
    # print(path)
    flipped_write = tf.write_file(path, flipped_img)
    sess.run(flipped_write)
