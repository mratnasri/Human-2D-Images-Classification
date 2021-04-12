from glob import glob
from keras.preprocessing.image import img_to_array, load_img
import tensorflow.compat.v1 as tf
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
        name = fn.split("/")[-1]
        names.append(name)
    return images, names


images, names = load_dataset(all_images)


def noise_img(img):
    original_size = list(img.shape)
    x = img
    noise = tf.random_normal(shape=original_size, mean=0.0, stddev=1.0, dtype=tf.float32)
    output = tf.add(x, noise)
    tf.cast(output, dtype=tf.uint8)
    return output


sess.run(tf.initialize_all_variables())

for j in range(len(images)):
    noised = noise_img(images[j])
    noised = 255*noised/tf.math.reduce_max(noised)
    noised = tf.cast(noised, dtype=tf.uint8)
    noised_img = tf.image.encode_png(noised)
    path =  '../../dataAugmentation_noise/'+names[j].replace('.jpg', '_noise.png')
    print(path)
    noise_write = tf.write_file(path, noised_img)
    #print("done")
    sess.run(noise_write)
