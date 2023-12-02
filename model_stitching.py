from __future__ import print_function, division

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Add,
    Average,
    Input,
    BatchNormalization,
    Activation,
)
from tensorflow.keras.layers import (
    Convolution2D,
    MaxPooling2D,
    UpSampling2D,
    Convolution2DTranspose,
)
from tensorflow.keras import backend as K

# from tensorflow.keras.utils.np_utils import to_categorical
import tensorflow.keras.callbacks as callbacks
import tensorflow.keras.optimizers as optimizers

from advanced import HistoryCheckpoint, TensorBoardBatch

# import img_utils
from data_generator import read_img_dataset  # , DataGenerator, image_stitching_generator
import prepare_data as psd
from sklearn.model_selection import train_test_split
import constant as cfg

import numpy as np
import os
import time
import warnings

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


def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return -10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)


def psnr(y_true, y_pred):
    assert y_true.shape == y_pred.shape, (
        "Cannot calculate PSNR. Input shapes not same."
        " y_true shape = %s, y_pred shape = %s" % (str(y_true.shape), str(y_pred.shape))
    )

    return -10.0 * np.log10(np.mean(np.square(y_pred - y_true)))


def SSIMLoss(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1)


class BaseStitchingModel(object):
    def __init__(self, model_name):
        """
        Base model to provide a standard interface of adding Image Stiching models
        """
        self.shape = (None, None, None)
        self.model = None  # type: Model
        self.model_name = model_name
        # self.scale_factor = 1.0
        self.weight_path = None

    def create_model(
        self, height=32, width=32, channels=3, nb_camera=5, load_weights=False
    ) -> Model:
        """
        Subclass dependent implementation.
        """
        self.shape = (width, height, channels * nb_camera)

        init = Input(shape=self.shape)

        return init

    def fit(
        self,
        batch_size=32,
        nb_epochs=100,
        save_history=True,
        history_fn="Model History.txt",
    ) -> Model:
        """
        Standard method to train any of the models.
        """

        if self.model is None:
            self.create_model()

        callback_list = [
            callbacks.ModelCheckpoint(
                self.weight_path,
                monitor="val_PSNRLoss",
                save_best_only=True,
                mode="max",
                save_weights_only=True,
                verbose=2,
            )
        ]

        # Parameters
        params = {
            "dim": (self.shape[0], self.shape[1]),
            "batch_size": batch_size,
            "n_channels": self.shape[2],
            "shuffle": True,
        }

        print("*************", self.shape[0], self.shape[1])
        # Datasets
        config_data = psd.read_json_file(cfg.config_img_output)
        data_indexes = np.arange(config_data["total_samples"])
        train_indexes, test_indexes = train_test_split(data_indexes, test_size=0.10)
        samples_per_epoch = len(train_indexes)
        val_count = len(test_indexes)

        # Generators
        # training_generator = image_stitching_generator(train_indexes, config_data,
        #                                                callee="training_generator", **params)
        # validation_generator = image_stitching_generator(test_indexes, config_data,
        #                                                  callee="validation_generator", **params)
        # training_generator = DataGenerator(train_indexes, config_data, callee="training_generator", **params)
        # validation_generator = DataGenerator(test_indexes, config_data, callee="validation_generator", **params)

        training_generator = read_img_dataset(
            train_indexes, config_data, callee="training_generator", **params
        )
        validation_generator = read_img_dataset(
            test_indexes, config_data, callee="validation_generator", **params
        )

        if save_history:
            callback_list.append(HistoryCheckpoint(f"{cfg.log_dir}/{history_fn}"))

            if K.backend() == "tensorflow":
                log_dir = f"{cfg.log_dir}/{self.model_name}_logs/"
                tensorboard = TensorBoardBatch(log_dir, batch_size=batch_size)
                callback_list.append(tensorboard)

        print("Training model : %s" % self.__class__.__name__)

        self.model.fit(
            training_generator,
            epochs=nb_epochs,
            steps_per_epoch=samples_per_epoch // batch_size + 1,
            callbacks=callback_list,
            validation_data=validation_generator,
            validation_steps=val_count // batch_size + 1,
            use_multiprocessing=True,
            workers=2,
        )

        return self.model

    def evaluate(self, validation_dir):
        pass

    def stitch(
        self,
        img_path,
        out_file=None,
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

        # Destination path
        if out_file is None:
            # out_dirname = os.path.abspath(os.path.dirname(img_path[0]))
            img_pathname = os.path.splitext(os.path.basename(img_path[0]))
            out_dirname = os.path.abspath(
                os.path.join(os.path.dirname(img_path[0]), "../..")
            )
            out_dirname = os.path.join(out_dirname, "out_result")
            filename = os.path.join(
                out_dirname,
                "result_"
                + str(suffix)
                + time.strftime("_%Y%m%d-%H%M%S")
                + img_pathname[1],
            )
        else:
            filename = out_file

        print("Output Result File: %s" % filename)

        img_conv, h, w = self.__read_conv_img(img_path, scale_factor)
        img_conv = img_conv.transpose((0, 2, 1, 3)).astype(np.float32) / 255.0

        print("Convolution image data point ready to be used: ", img_conv.shape)

        model = self.create_model(height=h, width=w, load_weights=True)
        if verbose:
            print("Model loaded.")

        # Create prediction for image patches
        print("Starting the image stitching prediction")
        result = model.predict(img_conv, verbose=verbose)

        # Deprocess patches
        if verbose:
            print("De-processing images.")

        result = result.transpose((0, 2, 1, 3)).astype(np.float32) * 255.0

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
        cv2.imwrite(filename, result)

        if return_image:
            # Return the image without saving. Useful for testing images.
            return result

    def __read_conv_img(self, img_paths: list, scaled_factor):

        # img_dir = os.path.dirname(img_path)
        # files = []
        # exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]
        # for ext in exts:
        #     files.extend(glob.glob(os.path.join(img_dir, ext)))

        true_img = cv2.imread(img_paths[0])
        h, w = true_img.shape[0], true_img.shape[1]
        print("Original Shape: ", h, w)

        final_h, final_w = int(true_img.shape[0] / scaled_factor), int(
            true_img.shape[1] / scaled_factor
        )
        print("Final Shape After Resize", final_h, final_w)
        # true_img = cv2.resize(true_img, (final_h, final_w))

        X = np.zeros((1, final_h, final_w, 15))

        for img_idx, img_path in enumerate(img_paths):
            img = cv2.imread(img_path)  # pilmode='RGB'
            img[np.where((img == [255, 255, 255]).all(axis=2))] = [0, 0, 0]
            img = cv2.resize(img, (final_w, final_h))
            print("image shape %d" % img_idx, img.shape)
            j = 3 * img_idx

            X[0, :, :, j : (j + 3)] = img

        return X, final_h, final_w

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

        filename = os.path.join(
            out_dir, "result_" + str(suffix) + time.strftime("_%Y%m%d-%H%M%S") + ".jpg"
        )
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

        result = result.transpose((0, 2, 1, 3)).astype(np.float32) * 255.0

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
    def __init__(self):
        super(ResNetStitch, self).__init__("ResNetStitch")

        # Treat this model as a denoising auto encoder
        # Force the fit, evaluate and upscale methods to take special care about image shape
        # self.type_requires_divisible_shape = True
        # self.uses_learning_phase = False

        self.n = 64
        self.mode = 2

        self.weight_path = "weights/ResNetStitch.h5"

    def create_model(
        self, height=32, width=32, channels=3, nb_camera=5, load_weights=False
    ):
        init = super(ResNetStitch, self).create_model(
            height, width, channels, nb_camera, load_weights
        )

        x0 = Convolution2D(
            64, (3, 3), activation="relu", padding="same", name="sr_res_conv1"
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
            3, (3, 3), activation="linear", padding="same", name="sr_res_conv_final"
        )(x)

        model = Model(init, x)

        adam = optimizers.Adam(learning_rate=1e-3)
        model.compile(optimizer=adam, loss="mse", metrics=[PSNRLoss, SSIMLoss])
        if load_weights:
            print(f"Loading model weights at {self.weight_path}...")
            model.load_weights(self.weight_path, by_name=True)
        model.summary()

        self.model = model
        return model

    def _residual_block(self, ip, id):
        mode = False if self.mode == 2 else None
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        init = ip

        x = Convolution2D(
            64,
            (3, 3),
            activation="linear",
            padding="same",
            name="sr_res_conv_" + str(id) + "_1",
        )(ip)
        x = BatchNormalization(
            axis=channel_axis, name="sr_res_batchnorm_" + str(id) + "_1"
        )(x, training=mode)
        x = Activation("relu", name="sr_res_activation_" + str(id) + "_1")(x)

        x = Convolution2D(
            64,
            (3, 3),
            activation="linear",
            padding="same",
            name="sr_res_conv_" + str(id) + "_2",
        )(x)
        x = BatchNormalization(
            axis=channel_axis, name="sr_res_batchnorm_" + str(id) + "_2"
        )(x, training=mode)

        m = Add(name="sr_res_merge_" + str(id))([x, init])

        return m

    def _upscale_block(self, ip, id):
        init = ip

        channel_dim = 1 if K.image_data_format() == "channels_first" else -1
        channels = init.shape[channel_dim]  # init._keras_shape[channel_dim]

        # x = Convolution2D(256, (3, 3), activation="relu", padding='same', name='sr_res_upconv1_%d' % id)(init)
        # x = SubPixelUpscaling(r=2, channels=self.n, name='sr_res_upscale1_%d' % id)(x)
        x = UpSampling2D()(init)
        x = Convolution2D(
            self.n,
            (3, 3),
            activation="relu",
            padding="same",
            name="sr_res_filter1_%d" % id,
        )(x)

        # x = Convolution2DTranspose(channels, (4, 4), strides=(2, 2), padding='same', activation='relu',
        #                            name='upsampling_deconv_%d' % id)(init)

        return x

    def fit(
        self,
        batch_size=128,
        nb_epochs=100,
        save_history=True,
        history_fn="ResNetSR History.txt",
    ):
        super(ResNetStitch, self).fit(batch_size, nb_epochs, save_history, history_fn)


class ImageStitchingModel(BaseStitchingModel):
    def __init__(self):
        super(ImageStitchingModel, self).__init__("Image Stitching Model")

        self.f1 = 9
        self.f2 = 1
        self.f3 = 5

        self.n1 = 64
        self.n2 = 32

        self.weight_path = "weights/Stitching Weights.h5"
        # self.type_true_upscaling = True

    def create_model(
        self, height=32, width=32, channels=3, nb_camera=5, load_weights=False
    ):
        """
        Creates a model to be used to scale images of specific height and width.
        """
        init = super(ImageStitchingModel, self).create_model(
            height, width, channels, nb_camera, load_weights
        )

        x = Convolution2D(
            self.n1, (self.f1, self.f1), activation="relu", padding="same", name="level1"
        )(init)
        x = Convolution2D(
            self.n2, (self.f2, self.f2), activation="relu", padding="same", name="level2"
        )(x)

        out = Convolution2D(channels, (self.f3, self.f3), padding="same", name="output")(
            x
        )

        model = Model(init, out)

        adam = optimizers.Adam(learning_rate=1e-3)
        model.compile(optimizer=adam, loss="mse", metrics=[PSNRLoss, SSIMLoss])
        if load_weights:
            print(f"Loading model weights at {self.weight_path}...")
            model.load_weights(self.weight_path)
        model.summary()

        self.model = model
        return model

    def fit(
        self,
        batch_size=128,
        nb_epochs=100,
        save_history=True,
        history_fn="ISCNN History.txt",
    ):
        return super(ImageStitchingModel, self).fit(
            batch_size, nb_epochs, save_history, history_fn
        )


class ExpantionStitching(BaseStitchingModel):
    def __init__(self):
        super(ExpantionStitching, self).__init__("Expanded Image SR")

        self.f1 = 9
        self.f2_1 = 1
        self.f2_2 = 3
        self.f2_3 = 5
        self.f3 = 5

        self.n1 = 64
        self.n2 = 32

        self.weight_path = "weights/ExpantionStitchWeights.h5"

    def create_model(
        self, height=32, width=32, channels=3, nb_camera=5, load_weights=False
    ):
        """
        Creates a model to be used to scale images of specific height and width.
        """
        init = super(ExpantionStitching, self).create_model(
            height=height,
            width=width,
            channels=channels,
            nb_camera=nb_camera,
            load_weights=load_weights,
        )

        x = Convolution2D(
            self.n1, (self.f1, self.f1), activation="relu", padding="same", name="level1"
        )(init)

        x1 = Convolution2D(
            self.n2,
            (self.f2_1, self.f2_1),
            activation="relu",
            padding="same",
            name="lavel1_1",
        )(x)
        x2 = Convolution2D(
            self.n2,
            (self.f2_2, self.f2_2),
            activation="relu",
            padding="same",
            name="lavel1_2",
        )(x)
        x3 = Convolution2D(
            self.n2,
            (self.f2_3, self.f2_3),
            activation="relu",
            padding="same",
            name="lavel1_3",
        )(x)

        x = Average()([x1, x2, x3])

        out = Convolution2D(
            channels, (self.f3, self.f3), activation="relu", padding="same", name="output"
        )(x)

        model = Model(init, out)
        adam = optimizers.Adam(learning_rate=1e-3)
        model.compile(optimizer=adam, loss="mse", metrics=[PSNRLoss, SSIMLoss])
        if load_weights:
            print(f"Loading model weights at {self.weight_path}...")
            model.load_weights(self.weight_path)
        model.summary()

        self.model = model
        return model

    def fit(
        self,
        batch_size=128,
        nb_epochs=100,
        save_history=True,
        history_fn="ESRCNN History IS.txt",
    ):
        return super(ExpantionStitching, self).fit(
            batch_size, nb_epochs, save_history, history_fn
        )


class DenoisingAutoEncoderStitch(BaseStitchingModel):
    def __init__(self):
        super(DenoisingAutoEncoderStitch, self).__init__("Denoise AutoEncoder IS")

        self.n1 = 64
        self.n2 = 32

        self.weight_path = "weights/DenoiseAutoEncoderStitch.h5"

    def create_model(
        self, height=32, width=32, channels=3, nb_camera=5, load_weights=False
    ):
        """
        Creates a model to remove / reduce noise from upscaled images.
        """
        from keras.layers.convolutional import Deconvolution2D

        # Perform check that model input shape is divisible by 4
        init = super(DenoisingAutoEncoderStitch, self).create_model(
            height=height, width=width, channels=channels, load_weights=load_weights
        )

        # if K.image_dim_ordering() == "th":
        #     output_shape = (None, channels, width, height)
        # else:
        #     output_shape = (None, width, height, channels)
        output_shape = (None, width, height, channels)

        level1_1 = Convolution2D(self.n1, (3, 3), activation="relu", padding="same")(init)
        level2_1 = Convolution2D(self.n1, (3, 3), activation="relu", padding="same")(
            level1_1
        )

        level2_2 = Convolution2DTranspose(
            self.n1, (3, 3), activation="relu", padding="same"
        )(level2_1)
        level2 = Add()([level2_1, level2_2])

        level1_2 = Convolution2DTranspose(
            self.n1, (3, 3), activation="relu", padding="same"
        )(level2)
        level1 = Add()([level1_1, level1_2])

        decoded = Convolution2D(channels, (5, 5), activation="linear", padding="same")(
            level1
        )

        model = Model(init, decoded)
        adam = optimizers.Adam(learning_rate=1e-3)
        model.compile(optimizer=adam, loss="mse", metrics=[PSNRLoss, SSIMLoss])
        if load_weights:
            print(f"Loading model weights at {self.weight_path}...")
            model.load_weights(self.weight_path)
        model.summary()

        self.model = model
        return model

    def fit(
        self,
        batch_size=128,
        nb_epochs=100,
        save_history=True,
        history_fn="DSRCNN HistoryIS.txt",
    ):
        return super(DenoisingAutoEncoderStitch, self).fit(
            batch_size, nb_epochs, save_history, history_fn
        )


class DistilledResNetStitch(BaseStitchingModel):
    def __init__(self):
        super(DistilledResNetStitch, self).__init__("Distilled ResNet Stitching")

        # Treat this model as a denoising auto encoder
        # Force the fit, evaluate and upscale methods to take special care about image shape
        self.type_requires_divisible_shape = True
        self.uses_learning_phase = False

        self.n = 32
        self.mode = 2

        self.weight_path = "weights/DistilledResNetStitch.h5"

    def create_model(
        self, height=32, width=32, channels=3, nb_camera=5, load_weights=False
    ):
        # init = super(DistilledResNetStich, self).create_model(height, width, channels, load_weights, batch_size)
        init = super(DistilledResNetStitch, self).create_model(
            height=height, width=width, channels=channels, load_weights=load_weights
        )

        x0 = Convolution2D(
            self.n, (3, 3), activation="relu", padding="same", name="student_sr_res_conv1"
        )(init)

        x = self._residual_block(x0, 1)

        x = Add(name="student_residual")([x, x0])
        x = self._upscale_block(x, 1)

        x = Convolution2D(
            3,
            (3, 3),
            activation="linear",
            padding="same",
            name="student_sr_res_conv_final",
        )(x)

        model = Model(init, x)
        # dont compile yet
        if load_weights:
            print(f"Loading model weights at {self.weight_path}...")
            model.load_weights(self.weight_path, by_name=True)
        model.summary()

        self.model = model
        return model

    def _residual_block(self, ip, id):
        mode = False if self.mode == 2 else None
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        init = ip

        x = Convolution2D(
            self.n,
            (3, 3),
            activation="linear",
            padding="same",
            name="student_sr_res_conv_" + str(id) + "_1",
        )(ip)
        x = BatchNormalization(
            axis=channel_axis, name="student_sr_res_batchnorm_" + str(id) + "_1"
        )(x, training=mode)
        x = Activation("relu", name="student_sr_res_activation_" + str(id) + "_1")(x)

        x = Convolution2D(
            self.n,
            (3, 3),
            activation="linear",
            padding="same",
            name="student_sr_res_conv_" + str(id) + "_2",
        )(x)
        x = BatchNormalization(
            axis=channel_axis, name="student_sr_res_batchnorm_" + str(id) + "_2"
        )(x, training=mode)

        m = Add(name="student_sr_res_merge_" + str(id))([x, init])

        return m

    def _upscale_block(self, ip, id):
        init = ip

        channel_dim = 1 if K.image_data_format() == "channels_first" else -1
        channels = init.shape[channel_dim]

        x = UpSampling2D(name="student_upsampling_%d" % id)(init)
        x = Convolution2D(
            self.n * 2,
            (3, 3),
            activation="relu",
            padding="same",
            name="student_sr_res_filter1_%d" % id,
        )(x)

        return x

    def fit(
        self,
        batch_size=128,
        nb_epochs=100,
        save_history=True,
        history_fn="DistilledResNetStitchHistory.txt",
    ):
        super(DistilledResNetStitch, self).fit(
            batch_size, nb_epochs, save_history, history_fn
        )


class DeepDenoiseStitch(BaseStitchingModel):
    def __init__(self):
        super(DeepDenoiseStitch, self).__init__("DeepDenoiseStitch")

        # Treat this model as a denoising auto encoder
        # Force the fit, evaluate and upscale methods to take special care about image shape
        self.type_requires_divisible_shape = True

        self.n1 = 64
        self.n2 = 128
        self.n3 = 256

        self.weight_path = "weights/DeepDenoiseStitch.h5"

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

        # decoded = Convolution2D(channels, 5, 5, activation='linear', padding='same')(m2)
        decoded = Convolution2D(channels, (5, 5), activation="linear", padding="same")(m2)

        model = Model(init, decoded)
        adam = optimizers.Adam(learning_rate=1e-3)
        model.compile(optimizer=adam, loss="mse", metrics=[PSNRLoss, SSIMLoss])
        if load_weights:
            print(f"Loading model weights at {self.weight_path}...")
            model.load_weights(self.weight_path)
        model.summary()

        self.model = model
        return model

    def fit(
        self,
        batch_size=128,
        nb_epochs=100,
        save_history=True,
        history_fn="DeepDenoiseStichHistory.txt",
    ):
        super(DeepDenoiseStitch, self).fit(
            batch_size, nb_epochs, save_history, history_fn
        )
