import scipy.misc
import numpy as np
import os, random
from glob import glob
import pickle

import tensorflow as tf
import tensorflow.contrib.slim as slim
from keras.datasets import cifar10, mnist


class ImageData:

    def __init__(self, load_size, channels, augment_flag):
        self.load_size = load_size
        self.channels = channels
        self.augment_flag = augment_flag

    def image_processing(self, filename):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels)
        img = tf.image.resize_images(x_decode, [self.load_size, self.load_size])
        img = tf.cast(img, tf.float32) / 127.5 - 1

        if self.augment_flag:
            augment_size = self.load_size + (30 if self.load_size == 256 else 15)
            p = random.random()
            if p > 0.5:
                img = augmentation(img, augment_size)

        return img


def augmentation(image, augment_size):
    seed = random.randint(0, 2 ** 31 - 1)
    ori_image_shape = tf.shape(image)
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.resize_images(image, [augment_size, augment_size])
    image = tf.random_crop(image, ori_image_shape, seed=seed)
    return image


def load_mnist(size=64):
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    train_data = normalize(train_data)
    test_data = normalize(test_data)

    x = np.concatenate((train_data, test_data), axis=0)
    # y = np.concatenate((train_labels, test_labels), axis=0).astype(np.int)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(x)
    # np.random.seed(seed)
    # np.random.shuffle(y)
    # x = np.expand_dims(x, axis=-1)

    x = np.asarray([scipy.misc.imresize(x_img, [size, size]) for x_img in x])
    x = np.expand_dims(x, axis=-1)
    return x


def load_cifar10(size=64):
    (train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
    train_data = normalize(train_data)
    test_data = normalize(test_data)

    x = np.concatenate((train_data, test_data), axis=0)
    # y = np.concatenate((train_labels, test_labels), axis=0).astype(np.int)

    seed = 777
    np.random.seed(seed)
    np.random.shuffle(x)
    # np.random.seed(seed)
    # np.random.shuffle(y)

    x = np.asarray([scipy.misc.imresize(x_img, [size, size]) for x_img in x])

    return x


def load_data(dataset_name, size=64):
    if dataset_name == 'mnist':
        x = load_mnist(size)
    elif dataset_name == 'cifar10':
        x = load_cifar10(size)
    else:

        x = glob(os.path.join("../../ImageNet40C_A", dataset_name, '*.*'))

    return x


def load_conv_autoe_features(y_dim, dataset_name):
    pickle_in = open("/data/ahmedfares/py/GANs/Keras_gans/autoencoder/conv_autoe_features_aug_2.pickle", "rb")
    conv_autoe_features = pickle.load(pickle_in)
    pickle_in.close()
    conv_autoe_features = conv_autoe_features.reshape((-1, 1, 1, y_dim))  # _size,128

    if dataset_name == 'n02106662':
        temp = conv_autoe_features[0:1300]
    elif dataset_name == 'n02124075':
        temp = conv_autoe_features[1300:1300 * 2]
    elif dataset_name == 'n02281787':
        temp = conv_autoe_features[1300 * 2:1300 * 3]
    elif dataset_name == 'n02389026':
        temp = conv_autoe_features[1300 * 3:1300 * 4]
    elif dataset_name == 'n02492035':
        temp = conv_autoe_features[1300 * 4:1300 * 5]
    elif dataset_name == 'n02504458':
        temp = conv_autoe_features[1300 * 5:1300 * 6]
    elif dataset_name == 'n02510455':
        temp = conv_autoe_features[1300 * 6:1300 * 7]
    elif dataset_name == 'n02607072':
        temp = conv_autoe_features[1300 * 7:1300 * 8]
    elif dataset_name == 'n02690373':
        temp = conv_autoe_features[1300 * 8:1300 * 9]
    elif dataset_name == 'n02906734':
        temp = conv_autoe_features[1300 * 9:1300 * 10]

    elif dataset_name == 'n02951358':
        temp = conv_autoe_features[1300 * 10:1300 * 11]
    elif dataset_name == 'n02992529':
        temp = conv_autoe_features[1300 * 11:1300 * 12]
    elif dataset_name == 'n03063599':
        temp = conv_autoe_features[1300 * 12:1300 * 13]
    elif dataset_name == 'n03100240':
        temp = conv_autoe_features[1300 * 13:1300 * 14]
    elif dataset_name == 'n03180011':
        temp = conv_autoe_features[1300 * 14:1300 * 15]
    elif dataset_name == 'n03197337':
        temp = conv_autoe_features[1300 * 15:1300 * 15 + 889]
    elif dataset_name == 'n03272010':
        temp = conv_autoe_features[1300 * 15 + 889:1300 * 15 + 889 + 1300]
    elif dataset_name == 'n03272562':
        temp = conv_autoe_features[1300 * 15 + 889 + 1300:1300 * 15 + 889 + 1300 * 2]
    elif dataset_name == 'n03297495':
        temp = conv_autoe_features[1300 * 15 + 889 + 1300 * 2:1300 * 15 + 889 + 1300 * 2 + 1136]
    elif dataset_name == 'n03376595':
        temp = conv_autoe_features[1300 * 15 + 889 + 1300 * 2 + 1136:1300 * 15 + 889 + 1300 * 2 + 1136 + 1300]

    elif dataset_name == 'n03445777':
        temp = conv_autoe_features[
               1300 * 15 + 889 + 1300 * 2 + 1136 + 1300:1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 2]
    elif dataset_name == 'n03452741':
        temp = conv_autoe_features[
               1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 2:1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 3]
    elif dataset_name == 'n03584829':
        temp = conv_autoe_features[
               1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 3:1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 4]
    elif dataset_name == 'n03590841':
        temp = conv_autoe_features[
               1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 4:1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 5]
    elif dataset_name == 'n03709823':
        temp = conv_autoe_features[
               1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 5:1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 6]
    elif dataset_name == 'n03773504':
        temp = conv_autoe_features[
               1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 6:1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 7]
    elif dataset_name == 'n03775071':
        temp = conv_autoe_features[
               1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 7:1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 8]
    elif dataset_name == 'n03792782':
        temp = conv_autoe_features[
               1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 8:1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 9]
    elif dataset_name == 'n03792972':
        temp = conv_autoe_features[
               1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 9:1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 10]
    elif dataset_name == 'n03877472':
        temp = conv_autoe_features[
               1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 10:1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 11]

    if dataset_name == 'n03888257':
        temp = conv_autoe_features[
               1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 11:1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 12]
    elif dataset_name == 'n03982430':
        temp = conv_autoe_features[
               1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 12:1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 13]
    elif dataset_name == 'n04044716':
        temp = conv_autoe_features[
               1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 13:1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 14]
    elif dataset_name == 'n04069434':
        temp = conv_autoe_features[
               1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 14:1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 15]
    elif dataset_name == 'n04086273':
        temp = conv_autoe_features[
               1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 15:1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 16]
    elif dataset_name == 'n04120489':
        temp = conv_autoe_features[
               1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 16:1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 17]
    elif dataset_name == 'n07753592':
        temp = conv_autoe_features[
               1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 17:1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 18]
    elif dataset_name == 'n07873807':
        temp = conv_autoe_features[
               1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 18:1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 19]
    elif dataset_name == 'n11939491':
        temp = conv_autoe_features[
               1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 19:1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 20]
    elif dataset_name == 'n13054560':
        temp = conv_autoe_features[
               1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 20:1300 * 15 + 889 + 1300 * 2 + 1136 + 1300 * 21]
    return temp


def preprocessing(x, size):
    x = scipy.misc.imread(x, mode='RGB')
    x = scipy.misc.imresize(x, [size, size])
    x = normalize(x)
    return x


def normalize(x):
    return x / 127.5 - 1


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
    # image = np.squeeze(merge(images, size)) # 채널이 1인거 제거 ?
    return scipy.misc.imsave(path, merge(images, size))


def inverse_transform(images):
    return (images + 1.) / 2.


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def str2bool(x):
    return x.lower() in ('true')

def save_images_gen(images, image_path):
    images = (images + 1.) / 2.

    for idx, image in enumerate(images):
        scipy.misc.imsave(image_path+'_idx_{}.png'.format(idx), image)
