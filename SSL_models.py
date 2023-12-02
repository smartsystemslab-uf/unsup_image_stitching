from __future__ import division, print_function

import os
import warnings
import shutil
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Convolution2D,
    Dropout,
    Input,
    # InstanceNormalization,
    LeakyReLU,
    MaxPooling2D,
    UpSampling2D,
)
from tensorflow.keras.models import Model

import constant as cfg

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

try:
    import cv2

    _cv2_available = True
except:
    warnings.warn(
        "Could not load opencv properly. This may affect the quality of output images."
    )
    _cv2_available = False


class VGGLossNetwork(tf.keras.models.Model):
    def __init__(
        self,
        style_layers=["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3"],
    ):
        super(VGGLossNetwork, self).__init__()
        vgg = tf.keras.applications.vgg16.VGG16(include_top=False, weights="imagenet")
        vgg.trainable = False
        model_outputs = [vgg.get_layer(name).output for name in style_layers]
        self.model = tf.keras.models.Model(vgg.input, model_outputs)
        # mixed precision float32 output
        self.linear = Activation("linear", dtype="float32")

    def call(self, x):
        x = tf.keras.applications.vgg16.preprocess_input(x)
        x = self.model(x)
        return self.linear(x)


class BaseStitchingModel(object):
    def __init__(self, model_name):
        """
        Base model to provide a standard interface of adding Image Stiching models
        """
        self.shape = (None, None, None)
        self.model = None  # type: Model
        self.model_name = model_name
        self.weight_path = None

    def create_model(
        self,
        height=32,
        width=32,
        channels=3,
        nb_camera=5,
        load_weights=False,
        train_mode=True,
    ) -> Model:
        """
        Subclass dependent implementation.
        """
        self.shape = (width, height, channels * nb_camera)

        init = Input(shape=self.shape, name="main_input")

        return init

    def deprocess(self, img):
        if self.activation == "tanh":
            return (img + 1.0) / 2.0
        else:
            return img

    def simple_stitch(
        self,
        img_conv,
        out_dir,
        suffix=None,
        return_image=False,
        scale_factor=1,
        verbose=True,
    ):
        """
        Standard method to upscale an image.
        :param img_path: list of path to input images
        :param out_file: Output folder to save all the results
        :param suffix: Suffix the be added to the output filename
        :param return_image: returns a image of shape (height, width, channels).
        :param scale_factor: image scaled factor to resize input images
        :param verbose: whether to print messages
        :param mode: mode of upscaling. Can be "patch" or "fast"
        """

        filename = os.path.join(out_dir, "result_" + str(suffix) + ".jpg")
        print("Output Result File: %s" % filename)
        os.makedirs(out_dir, exist_ok=True)

        # img_conv, h, w = self.__read_conv_img(img_path, scale_factor)
        h, w = img_conv.shape[1], img_conv.shape[2]
        img_conv = img_conv.transpose((0, 2, 1, 3))  # .astype(np.float32)

        print("Convolution image data point ready to be used: ", img_conv.shape)

        if not self.model:
            self.model = self.create_model(height=h, width=w, load_weights=True)
            if verbose:
                print("Model loaded.")

        # Create prediction for image patches
        print("Starting the image stitching prediction")
        result = self.model.predict(
            img_conv, verbose=verbose, workers=2, use_multiprocessing=True
        )

        # Deprocess patches
        if verbose:
            print("De-processing images.")

        result = (
            self.deprocess(result.transpose((0, 2, 1, 3)).astype(np.float32)) * 255.0
        )

        result = result[0, :, :, :]  # Access the 3 Dimensional image vector

        result = np.clip(result, 0, 255).astype("uint8")

        if _cv2_available:
            # used to remove noisy edges
            result = cv2.pyrUp(result)
            result = cv2.medianBlur(result, 3)
            result = cv2.pyrDown(result)

        if verbose:
            print("\nCompleted De-processing image.")

        if verbose:
            print("Saving image.", filename)
        # Convert into BGR to save with OpenCV
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, result)

        if return_image:
            # Return the image without saving. Useful for testing images.
            return result


class ResNetStitch(BaseStitchingModel):
    def __init__(self, metric=None):
        super(ResNetStitch, self).__init__("UnResNetStitch")

        self.n = 64
        # self.mode = 2

        self.weight_path = "weights/UnResNetStitch.h5"
        if metric and len(metric.strip()):
            self.weight_path = self.weight_path.replace(".h5", f"_{metric}.h5")

        self.model = None  # type: Model
        self.activation = (
            "tanh"  # The activation function of the last layer of the model
        )

    def create_model(
        self,
        height=32,
        width=32,
        channels=3,
        nb_camera=5,
        load_weights=False,
        train_mode=True,
    ):
        init = super(ResNetStitch, self).create_model(
            height, width, channels, nb_camera, load_weights, train_mode
        )

        x0 = Convolution2D(
            64,
            (3, 3),
            activation="relu",
            padding="same",
            name="sr_res_conv1",
            kernel_initializer="he_normal",
        )(init)

        x = self._residual_block(x0, 1)

        nb_residual = 5
        for i in range(nb_residual):
            x = self._residual_block(x, i + 2)

        x = Add()([x, x0])

        # x = self._upscale_block(x, 1)
        # x = Add()([x, x1])

        # x = self._upscale_block(x, 2)
        # x = Add()([x, x0])

        x = Convolution2D(
            3,
            (3, 3),
            activation=self.activation,
            padding="same",
            name="st_conv_final",
            kernel_initializer="he_normal",
        )(x)

        # m_custom_loss = gradient_layer_loss()([init, x])
        model = Model(init, x)
        model.summary()

        if load_weights and os.path.exists(self.weight_path):
            print(f"Loading model weights at {self.weight_path}...")
            model.load_weights(self.weight_path, by_name=True)
        elif load_weights:
            cfg.PRINT_WARNING(
                f"Cannot load the file {self.weight_path}, it doesn't exist!"
            )
        # model.summary()

        self.model = model
        return model

    def _residual_block(self, ip, id):
        # mode = True # False if self.mode == 2 else None
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        init = ip

        x = Convolution2D(
            64,
            (3, 3),
            activation="linear",
            padding="same",
            name="sr_res_conv_" + str(id) + "_1",
            kernel_initializer="he_normal",
        )(ip)
        x = BatchNormalization(
            axis=channel_axis, name="sr_res_batchnorm_" + str(id) + "_1"
        )(x)
        # x = Activation('relu', name="sr_res_activation_" + str(id) + "_1")(x)
        x = LeakyReLU(alpha=0.2, name="sr_res_activation_" + str(id) + "_1")(x)

        x = Convolution2D(
            64,
            (3, 3),
            activation="linear",
            padding="same",
            name="sr_res_conv_" + str(id) + "_2",
            kernel_initializer="he_normal",
        )(x)
        x = BatchNormalization(
            axis=channel_axis, name="sr_res_batchnorm_" + str(id) + "_2"
        )(x)
        # x = LeakyReLU(alpha=0.2, name="sr_res_activation_" + str(id) + "_2")(x)

        m = Add(name="sr_res_merge_" + str(id))([x, init])

        return m


class DeepDenoiseStitch(BaseStitchingModel):
    def __init__(self, metric=None):
        super(DeepDenoiseStitch, self).__init__("UnDeepDenoiseStitch")

        # Treat this model as a denoising auto encoder
        # Force the fit, evaluate and upscale methods to take special care about image shape

        self.n1 = 64
        self.n2 = 128
        self.n3 = 256

        self.weight_path = "weights/UnDeepDenoiseStitch.h5"
        if metric and len(metric.strip()):
            self.weight_path = self.weight_path.replace(".h5", f"_{metric}.h5")

        self.activation = (
            "tanh"  # The activation function of the last layer of the model
        )

    def create_model(
        self, height=32, width=32, channels=3, nb_camera=5, load_weights=False
    ):
        # Perform check that model input shape is divisible by 4

        init = super(DeepDenoiseStitch, self).create_model(
            height=height, width=width, channels=channels, load_weights=load_weights
        )

        c1 = Convolution2D(self.n1, (3, 3), activation="relu", padding="same")(init)
        c1 = Convolution2D(self.n1, (3, 3), activation="relu", padding="same")(c1)

        x = MaxPooling2D((2, 2))(c1)

        c2 = Convolution2D(self.n2, (3, 3), activation="relu", padding="same")(x)
        c2 = Convolution2D(self.n2, (3, 3), activation="relu", padding="same")(c2)

        x = MaxPooling2D((2, 2))(c2)

        c3 = Convolution2D(self.n3, (3, 3), activation="relu", padding="same")(x)

        x = UpSampling2D()(c3)

        c2_2 = Convolution2D(self.n2, (3, 3), activation="relu", padding="same")(x)
        c2_2 = Convolution2D(self.n2, (3, 3), activation="relu", padding="same")(c2_2)

        m1 = Add()([c2, c2_2])
        m1 = UpSampling2D()(m1)

        c1_2 = Convolution2D(self.n1, (3, 3), activation="relu", padding="same")(m1)
        c1_2 = Convolution2D(self.n1, (3, 3), activation="relu", padding="same")(c1_2)

        m2 = Add()([c1, c1_2])

        decoded = Convolution2D(
            channels,
            (5, 5),
            activation=self.activation,
            padding="same",
            name="st_conv_final",
        )(m2)

        model = Model(init, decoded)
        model.summary()

        if load_weights:
            if os.path.exists(self.weight_path):
                print(f"Loading model weights at {self.weight_path}...")
                model.load_weights(self.weight_path, by_name=True)
                dir_name, base_name = os.path.split(self.weight_path)
                backup_filename = (
                    f"backup_{datetime.now().strftime('%Y%m%d')}_{base_name}"
                )
                shutil.copy(self.weight_path, os.path.join(dir_name, backup_filename))
                print(f"Backup saved as {os.path.join(dir_name, backup_filename)}")
            else:
                cfg.PRINT_WARNING(
                    f"Cannot load the file {self.weight_path}, it doesn't exist!"
                )

        self.model = model
        return model


class U_NetStitch(BaseStitchingModel):
    def __init__(self, metric=None):
        super(U_NetStitch, self).__init__("U_NetStitch")

        # Treat this model as a denoising auto encoder
        # Force the fit, evaluate and upscale methods to take special care about image shape

        self.nb_filters = 64

        self.weight_path = "weights/U_NetStitch.h5"
        if metric and len(metric.strip()):
            self.weight_path = self.weight_path.replace(".h5", f"_{metric}.h5")

        self.activation = (
            "tanh"  # The activation function of the last layer of the model
        )

    def create_model(
        self, height=32, width=32, channels=3, nb_camera=5, load_weights=False
    ):
        # U-Net model Down sampling
        def down_sampling(
            x, filters, kernel_size=(4, 4), padding="same", strides=2, name=""
        ):
            d = Convolution2D(
                filters, kernel_size, padding=padding, strides=strides, name=name
            )(x)
            d = BatchNormalization()(d)
            # d = InstanceNormalization(axis=-1, center=False, scale=False)(d)
            d = Activation("relu")(d)
            return d

        # U-Net model Up sampling
        def up_sampling(
            x,
            skip,
            filters,
            kernel_size=(4, 4),
            padding="same",
            strides=1,
            dropout_rate=0,
            name="",
        ):
            c = UpSampling2D(size=2)(x)
            c = Convolution2D(
                filters, kernel_size, padding=padding, strides=strides, name=name
            )(c)
            c = BatchNormalization()(c)
            # c = InstanceNormalization(axis=-1, center=False, scale=False)(c)
            c = Activation("relu")(c)
            if dropout_rate:
                c = Dropout(dropout_rate)(c)
            c = Concatenate()([c, skip])
            return c

        # Input layer of the model
        init = super(U_NetStitch, self).create_model(
            height=height,
            width=width,
            channels=channels,
            nb_camera=nb_camera,
            load_weights=load_weights,
        )

        # U-Net model Down sampling
        d1 = down_sampling(init, self.nb_filters)
        d2 = down_sampling(d1, self.nb_filters * 2)
        d3 = down_sampling(d2, self.nb_filters * 4)
        d4 = down_sampling(d3, self.nb_filters * 8)

        # U-Net model Up sampling
        u1 = up_sampling(d4, d3, self.nb_filters * 4)
        u2 = up_sampling(u1, d2, self.nb_filters * 2)
        u3 = up_sampling(u2, d1, self.nb_filters)
        u4 = UpSampling2D(size=2)(u3)

        decoded = Convolution2D(
            channels,
            kernel_size=4,
            strides=1,
            activation=self.activation,
            padding="same",
            name="conv_final",
        )(u4)

        model = Model(init, decoded, name="U_NetStitch")
        model.summary()

        if load_weights and os.path.exists(self.weight_path):
            print(f"Loading model weights at {self.weight_path}...")
            model.load_weights(self.weight_path, by_name=True)
        elif load_weights:
            cfg.PRINT_WARNING(
                f"Cannot load the file {self.weight_path}, it doesn't exist!"
            )

        self.model = model
        return model
