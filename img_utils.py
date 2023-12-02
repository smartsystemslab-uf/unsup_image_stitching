from __future__ import print_function, division, absolute_import

import numpy as np

# from scipy.misc import imsave, imread, imresize
from imageio import imwrite, imread
from cv2 import resize, INTER_CUBIC
from sklearn.feature_extraction.image import (
    reconstruct_from_patches_2d,
    extract_patches_2d,
)
from scipy.ndimage.filters import gaussian_filter

import patchify

from tensorflow.keras import backend as K

import os
import time

import tensorflow as tf
import numpy as np
import PIL.Image
import os
import json
import prepare_data as psd
import constant as cfg

hparams = {
    "content_weight": 1e-1,  # 1e-5,
    "style_weight": 4e-9,
    "simple_weight": 2e0,  # 4e-5, # 4e-9,
    "gradient_weight": 4e2,  # 4e-1,
    "tv_weight": 4e-5,  # 4e-7,
    "learning_rate": 0.001,
    "residual_filters": 128,
    "residual_layers": 5,
    "initializer": "glorot_normal",
    "style_layers": ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3"],
    "content_layer_index": 2,
}


def deprocess(img):
    img = 255 * (img + 1.0) / 2.0
    return tf.cast(img, tf.uint8)


def convert(file_path, shape):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, shape)
    return img


def tensor_to_image(tensor):
    tensor = 255 * (tensor + 1.0) / 2.0
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def gram_matrix(input_tensor):
    input_tensor = tf.cast(input_tensor, tf.float32)  # avoid mixed_precision nan
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(
        input_shape[1] * input_shape[2], tf.float32
    )  # int32 to float32
    return result / num_locations


def style_loss(target_feature_maps, output_feature_maps):

    # Get output gram_matrix
    target_gram_matrices = [gram_matrix(x) for x in target_feature_maps]
    output_gram_matrices = [gram_matrix(x) for x in output_feature_maps]
    num_style_layers = len(target_feature_maps)
    return (
        tf.add_n(
            [
                tf.reduce_mean((style_feat - out_feat) ** 2)
                for style_feat, out_feat in zip(
                    target_gram_matrices, output_gram_matrices
                )
            ]
        )
        / num_style_layers
    )


def content_loss(content, output):
    return tf.reduce_mean((content - output) ** 2)


def total_variation_loss(content):
    a = K.square(content[:, :-1, :-1, :] - content[:, 1:, :-1, :])
    b = K.square(content[:, :-1, :-1, :] - content[:, :-1, 1:, :])
    loss = K.mean(K.sum(K.pow(a + b, 1.25)))
    return loss


def gradient_loss(content, output):
    "Ensure the smoothness of the images"

    dy_content, dx_content = tf.image.image_gradients(content)
    dy_ouput, dx_ouput = tf.image.image_gradients(output)

    ly = (dy_content - dy_ouput) ** 2
    lx = (dx_content - dx_ouput) ** 2

    return tf.reduce_mean(lx + ly)


def save_hparams(model_name):
    json_hparams = json.dumps(hparams)
    f = open(os.path.join(cfg.project_dir, "{}_hparams.json".format(model_name)), "w")
    f.write(json_hparams)
    f.close()


def save_dataset_indexes(
    filename_prefix, train_indexes, test_indexes, samples_per_epoch, val_count
):
    hindexses = {
        "train_indexes": train_indexes,
        "test_indexes": test_indexes,
        "samples_per_epoch": samples_per_epoch,
        "val_count": val_count,
    }
    filename = os.path.join(cfg.project_dir, "{}.json".format(filename_prefix))
    cfg.PRINT_INFO(f"Saving dataset indexes: {filename}")
    psd.write_json_file(filename, hindexses)


def get_dataset_indexes(filename_prefix):
    filename = os.path.join(cfg.project_dir, "{}.json".format(filename_prefix))
    cfg.PRINT_INFO(f"Reading dataset indexes: {filename} ...")
    config_data = psd.read_json_file(filename)
    return config_data


def delete_dataset_indexes(filename_prefix):
    filename = os.path.join(cfg.project_dir, "{}.json".format(filename_prefix))
    cfg.PRINT_INFO(f"Delete dataset indexes: {filename} ...")
    config_data = psd.remove_file(filename)


def make_patches(x, scale, patch_size, upscale=True, verbose=1):
    """x shape: (num_channels, rows, cols)"""
    height, width = x.shape[:2]
    if upscale:
        x = resize(x, (height * scale, width * scale))
    patches = extract_patches_2d(x, (patch_size, patch_size))
    return patches


def patchyfy(img, patch_shape, step=32):
    X, Y, c = img.shape
    x, y = patch_shape
    shape = (X - x + step, Y - y + step, x, y, c)
    X_str, Y_str, c_str = img.strides
    strides = (X_str, Y_str, X_str, Y_str, c_str)
    return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)


def make_raw_patches(x, patch_size, channels=3, step=1, verbose=1):
    """x shape: (rows, cols, num_channels)"""

    w_shape = x.shape - np.array([patch_size, patch_size, channels])
    if (w_shape < 0).any():
        r_plus = -1 * min(w_shape[0], 0)
        c_plus = -1 * min(w_shape[1], 0)
        x = np.pad(x, ((0, r_plus), (0, c_plus), (0, 0)), "symmetric")

    patches = patchify.patchify(x, (patch_size, patch_size, channels), step=step)
    return patches
