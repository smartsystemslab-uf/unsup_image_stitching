from __future__ import division, print_function

import argparse
import random
import time

import numpy as np
import tensorflow as tf
import tensorflow.keras.callbacks as callbacks
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model

import constant as cfg
import img_utils
import prepare_data as psd
from advanced import HistoryCheckpoint, TensorBoardBatch
from data_generator import read_img_dataset
from loss import ISMetric, LPIPSMetric, PNSRMetric, SSIMMetric
from model_stitching import DeepDenoiseStitch as DDStitch
from model_stitching import DenoisingAutoEncoderStitch as DAutoEncoderStitch
from model_stitching import DistilledResNetStitch as DResStitch
from model_stitching import ExpantionStitching as ExpStitch
from model_stitching import ImageStitchingModel as DPImgStitch
from model_stitching import ResNetStitch as ResNetStitch
from SSL_models import DeepDenoiseStitch as UnDDStitch
from SSL_models import ResNetStitch as UnResNetStitch
from SSL_models import VGGLossNetwork
from SSL_models import U_NetStitch
from un_data_generator import un_read_img_dataset

model_directory = {
    "DDStitch": DDStitch,
    "DResStitch": DResStitch,
    "ResNetStitch": ResNetStitch,
    "DPImgStitch": DPImgStitch,
    "ExpStitch": ExpStitch,
    "DAutoEncoderStitch": DAutoEncoderStitch,
    "UnResNetStitch": UnResNetStitch,
    "UnDDStitch": UnDDStitch,
    "U_NetStitch": U_NetStitch,
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_dir",
    default="experiments/base_model",
    help="Directory containing params.json",
)
parser.add_argument(
    "--model", default="DDStitch", help="Deep Denoise Stitching Model", type=str
)
parser.add_argument(
    "--load_weights",
    default=True,
    type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
)
parser.add_argument(
    "--save_model_img",
    default=True,
    type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
)
parser.add_argument(
    "--supervised", default=True, type=lambda x: (str(x).lower() in ["true", "1", "yes"])
)
parser.add_argument("--nb_epochs", default=10, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument(
    "--restore_file",
    default=None,
    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training",
)  # 'best' or 'train'

args = parser.parse_args()
print("==> Training Argument: ", args)
net = model_directory[args.model]
model_name = args.model

model = None
# from tensorflow.keras import mixed_precision
# policy = mixed_precision.Policy('float32')
# mixed_precision.set_global_policy(policy)


def print_status_bar(iteration, total, loss=None, metrics=None):
    metrics = " - ".join(
        ["{}: {:.4f}".format(m.name, m.result()) for m in (loss or []) + (metrics or [])]
    )
    end = "" if iteration < total else "\n"
    print("\r===> {}/{} - ".format(iteration, total) + metrics, end=end)


def train(height, width, nb_epochs=10, batch_size=32, save_arch=False, load_weights=True):
    global model

    # stitch_model = model_stitching.NonLocalResNetStitching()
    stitch_model = net()  # model_stitching.DeepDenoiseStitch()
    stitch_model.create_model(height=height, width=width, load_weights=load_weights)

    if save_arch:
        plot_model(
            stitch_model.model,
            to_file=f"architectures/model_img/{model_name}.png",
            show_shapes=True,
            show_layer_names=True,
        )

    model = stitch_model.model
    loss_network = VGGLossNetwork()

    callback_list = []

    # Parameters
    params = {
        "dim": (width, height),
        "batch_size": batch_size,
        "n_channels": 15,
        "shuffle": True,
    }
    history_fn = f"{model_name}History.txt"

    # Get Datasets
    file_index_pref = "sup_training_data_ids"
    h_indexes = img_utils.get_dataset_indexes(file_index_pref)
    config_data = psd.read_json_file(cfg.config_img_output)
    if h_indexes:
        train_indexes = np.array(h_indexes["train_indexes"])
        test_indexes = np.array(h_indexes["test_indexes"])
        samples_per_epoch = h_indexes["samples_per_epoch"]
        val_count = h_indexes["val_count"]
    else:
        data_indexes = np.arange(config_data["total_samples"])
        print(f"Total sample: {config_data['total_samples']}")
        train_indexes, test_indexes = train_test_split(data_indexes, test_size=0.10)
        samples_per_epoch = len(train_indexes)
        val_count = len(test_indexes)
        img_utils.save_dataset_indexes(
            file_index_pref,
            train_indexes.tolist(),
            test_indexes.tolist(),
            samples_per_epoch,
            val_count,
        )

    train_dataset = read_img_dataset(
        train_indexes, config_data, callee="training_generator", **params
    )
    val_dataset = read_img_dataset(
        test_indexes, config_data, callee="validation_generator", **params
    )

    callback_list.append(
        callbacks.ModelCheckpoint(
            stitch_model.weight_path,
            monitor="val_LPIPSMetric",  # "loss",
            save_best_only=True,
            mode="min",
            save_weights_only=True,
            save_freq="epoch",
            verbose=2,
        )
    )

    # TODO: Guard this with an if condition to use this checkpoint only when dealing with supervised training
    callback_list.append(
        callbacks.ModelCheckpoint(
            stitch_model.weight_path.replace(".h5", "_ssim.h5"),
            monitor="val_SSIMMetric",
            save_best_only=True,
            mode="max",
            save_weights_only=True,
            save_freq="epoch",
            verbose=2,
        )
    )

    callback_list.append(
        callbacks.ModelCheckpoint(
            stitch_model.weight_path.replace(".h5", "_is.h5"),
            monitor="val_ISMetric",
            save_best_only=True,
            mode="max",
            save_weights_only=True,
            save_freq="epoch",
            verbose=2,
        )
    )

    callback_list.append(
        callbacks.ModelCheckpoint(
            stitch_model.weight_path.replace(".h5", "_lpips.h5"),
            monitor="val_LPIPSMetric",
            save_best_only=True,
            mode="min",
            save_weights_only=True,
            save_freq="epoch",
            verbose=2,
        )
    )

    callback_list.append(
        callbacks.ModelCheckpoint(
            stitch_model.weight_path.replace(".h5", "_pnsr.h5"),
            monitor="val_PSNRMetric",
            save_best_only=True,
            mode="max",
            save_weights_only=True,
            save_freq="epoch",
            verbose=2,
        )
    )

    callback_list.append(HistoryCheckpoint(f"{cfg.log_dir}/{history_fn}"))
    log_dir = f"{cfg.log_dir}/{stitch_model.model_name}_logs/"
    tensorboard = TensorBoardBatch(log_dir, batch_size=batch_size)
    callback_list.append(tensorboard)
    all_callbacks = callbacks.CallbackList(callback_list, add_history=True, model=model)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6, epsilon=0.01)

    total_loss_avg = tf.keras.metrics.Mean(name="Total Loss")
    style_loss_avg = tf.keras.metrics.Mean(name="Style Loss")
    content_loss_avg = tf.keras.metrics.Mean(name="Content Loss")
    gradient_loss_avg = tf.keras.metrics.Mean(name="Gradient Loss")
    simple_loss_avg = tf.keras.metrics.Mean(name="Simple Loss (MSE)")
    tv_loss_avg = tf.keras.metrics.Mean(name="Avg. TV Loss")

    metrics = [
        PNSRMetric(name="val_PSNRMetric"),
        SSIMMetric(name="val_SSIMMetric"),
        LPIPSMetric(name="val_LPIPSMetric"),
        ISMetric(name="val_ISMetric", img_shape=(width, height, 3)),
    ]

    simple_loss_fn = tf.keras.losses.mean_squared_error

    logs = {}
    all_callbacks.on_train_begin(logs=logs)
    epochs = nb_epochs

    @tf.function
    def test_step(batch, return_dict=True):
        x_batch, y_batch = batch
        prediction = model(x_batch, training=False)
        y_pred_clip = (prediction + 1.0) / 2.0

        for metric in metrics:
            if metric.name == "val_ISMetric":
                metric(y_batch)
            else:
                metric(y_batch, y_pred_clip)

        if return_dict:
            for metric in metrics:
                logs[metric.name] = metric.result()

            return logs

        # return img_utils.deprocess(prediction)

    @tf.function
    def train_step(batch, return_dict=True):
        # model = Model
        x_batch, y_batch = batch
        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)
            y_pred_clip = (y_pred + 1.0) / 2.0
            output_batch = y_pred_clip  # float deprocess
            target_batch = y_batch  # float deprocess

            # Feed target and output batch through loss_network
            target_batch_feature_maps = loss_network(target_batch)
            output_batch_feature_maps = loss_network(output_batch)

            c_loss = img_utils.content_loss(
                target_batch_feature_maps[img_utils.hparams["content_layer_index"]],
                output_batch_feature_maps[img_utils.hparams["content_layer_index"]],
            )
            c_loss *= img_utils.hparams["content_weight"]
            # c_loss = tf.cond(tf.math.is_nan(c_loss), lambda: 0.0, lambda: c_loss)

            # s_loss = img_utils.style_loss(target_batch_feature_maps,
            #                     output_batch_feature_maps)
            # s_loss *= img_utils.hparams['style_weight']
            s_loss = 0

            grad_loss = img_utils.hparams["gradient_weight"] * img_utils.gradient_loss(
                y_batch, y_pred_clip
            )
            # grad_loss = tf.cond(tf.math.is_nan(grad_loss), lambda: 0.0, lambda: grad_loss)

            mse_loss = img_utils.hparams["simple_weight"] * tf.reduce_mean(
                simple_loss_fn(y_batch, y_pred_clip)
            )
            # mse_loss = tf.cond(tf.math.is_nan(mse_loss), lambda: 0.0, lambda: mse_loss)

            tv_loss = img_utils.hparams["tv_weight"] * img_utils.total_variation_loss(
                output_batch
            )
            # tv_loss = tf.cond(tf.math.is_nan(tv_loss), lambda: 0.0, lambda: tv_loss)

            main_loss = c_loss + s_loss + mse_loss + grad_loss + tv_loss
            total_loss = tf.add_n([main_loss] + model.losses)
        #     scaled_loss = optimizer.get_scaled_loss(total_loss)

        # scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
        # gradients = optimizer.get_unscaled_gradients(scaled_gradients)
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        total_loss_avg(total_loss)
        content_loss_avg(c_loss)
        style_loss_avg(s_loss)
        simple_loss_avg(mse_loss)
        gradient_loss_avg(grad_loss)
        tv_loss_avg(tv_loss)

        # for metric in metrics:
        #     metric(y_batch, y_pred_clip)

        if return_dict:
            logs = {
                "loss": total_loss_avg.result(),
                "total_loss": total_loss_avg.result(),
                "content_loss": content_loss_avg.result(),
                "style_loss": style_loss_avg.result(),
                "grad_loss": gradient_loss_avg.result(),
                "simple_loss": simple_loss_avg.result(),
                "tv_loss": tv_loss_avg.result(),
            }
            return logs

    steps_per_epoch = samples_per_epoch // batch_size
    validation_steps = val_count // batch_size
    total_time = time.time()

    all_callbacks.on_train_begin()
    for epoch in range(1, epochs + 1):
        all_callbacks.on_epoch_begin(epoch, logs=logs)
        print("Epoch {} / {}".format(epoch, epochs))
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            all_callbacks.on_train_batch_begin(step, logs=logs)

            logs = train_step(batch=(x_batch_train, y_batch_train))

            all_callbacks.on_train_batch_end(step, logs=logs)

            print_status_bar(
                step,
                steps_per_epoch,
                [
                    total_loss_avg,
                    style_loss_avg,
                    content_loss_avg,
                    gradient_loss_avg,
                    simple_loss_avg,
                    tv_loss_avg,
                ],
            )
            # break

        # Run a validation loop at the end of each epoch.
        for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
            all_callbacks.on_test_batch_begin(step, logs=logs)

            logs = test_step(batch=(x_batch_val, y_batch_val))

            all_callbacks.on_test_batch_end(step, logs=logs)

            print_status_bar(step, validation_steps, metrics=metrics)

        # print("Validation acc: %.4f" % (float(val_acc),))
        all_callbacks.on_epoch_end(epoch, logs=logs)

        print("\nTime taken per epoch: %.2fs" % (time.time() - start_time))
        for metric in [
            total_loss_avg,
            content_loss_avg,
            style_loss_avg,
            gradient_loss_avg,
            simple_loss_avg,
            tv_loss_avg,
        ] + metrics:
            metric.reset_states()

    print("\nTotal Time Training: %.2fs" % (time.time() - total_time))
    all_callbacks.on_train_end(logs=logs)
    img_utils.delete_dataset_indexes(file_index_pref)


def un_train(
    height, width, nb_epochs=10, batch_size=32, save_arch=False, load_weights=True
):
    """Unsupervised Training"""
    global model

    stitch_model = net()  # model_stitching.DeepDenoiseStitch()
    stitch_model.create_model(height=height, width=width, load_weights=load_weights)

    if save_arch:
        plot_model(
            stitch_model.model,
            to_file=f"architectures/model_img/{model_name}.png",
            show_shapes=True,
            show_layer_names=True,
        )

    model = stitch_model.model
    loss_network = VGGLossNetwork()

    callback_list = []

    # Parameters
    params = {
        "dim": (width, height),
        "batch_size": 1,  # Read one sample at time
        "buffer_size": batch_size,
        "n_channels": 15,
        "shuffle": True,
    }
    history_fn = f"{model_name}History.txt"

    # Get Datasets
    file_index_pref = "un_sup_training_data_ids"
    h_indexes = img_utils.get_dataset_indexes(file_index_pref)
    config_data = psd.read_json_file(cfg.un_config_img_output)
    if h_indexes:
        train_indexes = np.array(h_indexes["train_indexes"])
        test_indexes = np.array(h_indexes["test_indexes"])
        samples_per_epoch = h_indexes["samples_per_epoch"]
        val_count = h_indexes["val_count"]
    else:
        data_indexes = np.arange(config_data["total_samples"])
        print(f"Total sample: {config_data['total_samples']}")
        train_indexes, test_indexes = train_test_split(data_indexes, test_size=0.10)
        train_indexes = np.random.permutation(config_data["total_samples"])
        samples_per_epoch = len(train_indexes)
        val_count = len(test_indexes)
        img_utils.save_dataset_indexes(
            file_index_pref,
            train_indexes.tolist(),
            test_indexes.tolist(),
            samples_per_epoch,
            val_count,
        )
        # img_utils.save_dataset_indexes(file_index_pref, train_indexes.tolist(), "test_indexes.tolist()", samples_per_epoch, "val_count")

    train_dataset = un_read_img_dataset(
        train_indexes, config_data, callee="un_training_generator", **params
    )
    val_dataset = un_read_img_dataset(
        test_indexes, config_data, callee="un_validation_generator", **params
    )

    callback_list.append(
        callbacks.ModelCheckpoint(
            stitch_model.weight_path,
            monitor="val_LPIPSMetric",  # "loss",
            save_best_only=True,
            mode="min",
            save_weights_only=True,
            save_freq="epoch",
            verbose=2,
        )
    )

    # TODO: Guard this with an if condition to use this checkpoint only when dealing with supervised training
    # callback_list.append(callbacks.ModelCheckpoint(stitch_model.weight_path.replace(".h5", "_ssim.h5"),
    #                                         monitor='val_SSIMMetric', save_best_only=True,
    #                                         mode='max', save_weights_only=True,
    #                                         save_freq='epoch', verbose=2))

    callback_list.append(
        callbacks.ModelCheckpoint(
            stitch_model.weight_path.replace(".h5", "_is.h5"),
            monitor="val_ISMetric",
            save_best_only=True,
            mode="max",
            save_weights_only=True,
            save_freq="epoch",
            verbose=2,
        )
    )

    callback_list.append(
        callbacks.ModelCheckpoint(
            stitch_model.weight_path.replace(".h5", "_lpips.h5"),
            monitor="val_LPIPSMetric",
            save_best_only=True,
            mode="min",
            save_weights_only=True,
            save_freq="epoch",
            verbose=2,
        )
    )

    # callback_list.append(callbacks.ModelCheckpoint(stitch_model.weight_path.replace(".h5", "_pnsr.h5"),
    #                                         monitor='val_PSNRMetric', save_best_only=True,
    #                                         mode='max', save_weights_only=True,
    #                                         save_freq='epoch', verbose=2))

    callback_list.append(HistoryCheckpoint(f"{cfg.log_dir}/{history_fn}"))
    log_dir = f"{cfg.log_dir}/{stitch_model.model_name}_logs/"
    tensorboard = TensorBoardBatch(log_dir, batch_size=batch_size)
    callback_list.append(tensorboard)
    all_callbacks = callbacks.CallbackList(callback_list, add_history=True, model=model)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    total_loss_avg = tf.keras.metrics.Mean(name="Avg. Total Loss")
    style_loss_avg = tf.keras.metrics.Mean(name="Avg. Style Loss")
    content_loss_avg = tf.keras.metrics.Mean(name="Avg. Content Loss")
    tv_loss_avg = tf.keras.metrics.Mean(name="Avg. TV Loss")
    # gradient_loss_avg = tf.keras.metrics.Mean(name='Gradient Loss')
    # simple_loss_avg = tf.keras.metrics.Mean(name='Simple Loss (MSE)')

    metrics = [
        LPIPSMetric(name="val_LPIPSMetric"),
        ISMetric(name="val_ISMetric", img_shape=(width, height, 3)),
    ]

    # simple_loss_fn = tf.keras.losses.mean_squared_error

    logs = {}
    all_callbacks.on_train_begin(logs=logs)
    epochs = nb_epochs

    def compute_loss(target_batch, output_batch):

        # Feed target and output batch through loss_network
        mask = tf.where(target_batch != 0.0, 1.0, 0.0)
        target_batch_feature_maps = loss_network(target_batch)
        output_batch_mask = output_batch * mask

        output_batch_feature_maps = loss_network(output_batch_mask)
        # num_style_layers = len(target_batch_feature_maps)

        c_loss = img_utils.content_loss(
            target_batch_feature_maps[img_utils.hparams["content_layer_index"]],
            output_batch_feature_maps[img_utils.hparams["content_layer_index"]],
        )
        c_loss *= img_utils.hparams["content_weight"] / batch_size

        # Get output gram_matrix
        # s_loss = img_utils.style_loss(target_batch_feature_maps,  output_batch_feature_maps)
        # s_loss *= (img_utils.hparams['style_weight'] / batch_size)
        s_loss = 0

        tv_loss = img_utils.total_variation_loss(output_batch_mask)
        tv_loss *= img_utils.hparams["tv_weight"] / batch_size

        main_loss = c_loss + s_loss + tv_loss  # + mse_loss + grad_loss #
        total_loss = tf.add_n([main_loss] + model.losses)

        return total_loss, c_loss, s_loss, tv_loss

    # @tf.function
    def test_step(batch, return_dict=True):
        x_batch, y_batch = batch
        prediction = model(x_batch, training=False)
        y_pred_clip = (prediction + 1.0) / 2.0

        for metric in metrics:

            nb_cameras = tf.cast(tf.cast(y_batch.shape[-1], tf.float32) / 3.0, tf.uint8)
            r = list(range(nb_cameras))
            # random.shuffle(r)
            for i in r:
                mask = tf.where(y_batch[:, :, :, i * 3 : (i + 1) * 3] != 0.0, 1.0, 0.0)
                if metric.name == "val_ISMetric":
                    metric(y_pred_clip * mask)
                else:
                    metric(y_batch[:, :, :, i * 3 : (i + 1) * 3], y_pred_clip * mask)

        if return_dict:
            for metric in metrics:
                logs[metric.name] = metric.result()

            return logs

    def train_step(sample_data, batch_id, batch_size, loss_dict, return_dict=True):
        # model = Model
        x_batch, y_batch = sample_data
        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)
            y_pred_clip = (y_pred + 1.0) / 2.0
            output_batch = y_pred_clip  # float deprocess
            target_batch = y_batch  # float deprocess
            # output_batch_feature_maps = loss_network(output_batch)
            # loss_dict = {"total_loss": 0, "st_loss": 0, "ct_loss": 0}
            # total_loss = loss_dict["total_loss"]
            # c_loss = loss_dict["st_loss"]
            # s_loss = loss_dict["ct_loss"]

            nb_cameras = tf.cast(
                tf.cast(target_batch.shape[-1], tf.float32) / 3.0, tf.uint8
            )
            r = list(range(nb_cameras))
            random.shuffle(r)
            for i in r:
                tl, cl, sl, tv = compute_loss(
                    target_batch=target_batch[:, :, :, i * 3 : (i + 1) * 3],
                    output_batch=output_batch,
                )
                loss_dict["total_loss"] += tl
                loss_dict["st_loss"] += sl
                loss_dict["ct_loss"] += cl
                loss_dict["tv_loss"] += tv

        # scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
        # gradients = optimizer.get_unscaled_gradients(scaled_gradients)
        if batch_id >= batch_size - 1:
            gradients = tape.gradient(loss_dict["total_loss"], model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        total_loss_avg(loss_dict["total_loss"])
        content_loss_avg(loss_dict["ct_loss"])
        style_loss_avg(loss_dict["st_loss"])
        tv_loss_avg(loss_dict["tv_loss"])

        if return_dict:
            logs = {
                "loss": total_loss_avg.result(),
                "total_loss": total_loss_avg.result(),
                "content_loss": content_loss_avg.result(),
                "style_loss": style_loss_avg.result(),
                "tv_loss": tv_loss_avg.result(),
            }
            return logs

    steps_per_epoch = samples_per_epoch // batch_size
    validation_steps = val_count // batch_size
    cfg.PRINT_INFO(
        f"validation_steps: {validation_steps}, val_count: {val_count}, batch_size: {batch_size}"
    )
    total_time = time.time()

    all_callbacks.on_train_begin()
    for epoch in range(1, epochs + 1):
        all_callbacks.on_epoch_begin(epoch, logs=logs)
        print("Epoch {} / {}".format(epoch, epochs))
        start_time = time.time()

        # Iterate over the batches of the dataset.
        batchid = 0
        batch_step = 0
        loss_dict = {"total_loss": 0, "st_loss": 0, "ct_loss": 0, "tv_loss": 0}
        for (x_batch_train, y_batch_train) in train_dataset:

            if batchid == 0:
                all_callbacks.on_train_batch_begin(batch_step, logs=logs)

            # logs = model.train_on_batch(x={"main_input": x_batch_train, "x_true_input": y_batch_train},
            # y={"st_conv_final": y_batch_train}, return_dict=True)
            logs = train_step(
                (x_batch_train, y_batch_train),
                batch_id=batchid,
                batch_size=batch_size,
                loss_dict=loss_dict,
            )
            batchid = batchid + 1

            if batchid >= batch_size:
                all_callbacks.on_train_batch_end(batch_step, logs=logs)
                print_status_bar(
                    batch_step,
                    steps_per_epoch,
                    [total_loss_avg, style_loss_avg, content_loss_avg, tv_loss_avg],
                )
                batchid = 0
                loss_dict = {"total_loss": 0, "st_loss": 0, "ct_loss": 0, "tv_loss": 0}
                batch_step += 1
            # break

        # Run a validation loop at the end of each epoch.
        batchid = 0
        batch_step = 0
        for (x_batch_val, y_batch_val) in val_dataset:
            if batchid == 0:
                all_callbacks.on_test_batch_begin(batch_step, logs=logs)

            logs = test_step(batch=(x_batch_val, y_batch_val))
            batchid = batchid + 1

            if batchid >= batch_size:
                all_callbacks.on_test_batch_end(batch_step, logs=logs)
                print_status_bar(batch_step, validation_steps, metrics=metrics)
                batchid = 0
                batch_step += 1

        all_callbacks.on_epoch_end(epoch, logs=logs)

        print("\nTime taken per epoch: %.2fs" % (time.time() - start_time))
        for metric in [total_loss_avg, content_loss_avg, style_loss_avg]:
            metric.reset_states()

    print("\nTotal Time Training: %.2fs" % (time.time() - total_time))
    all_callbacks.on_train_end(logs=logs)
    # img_utils.delete_dataset_indexes(file_index_pref)


def save_model_plots():
    for modname in model_directory:
        print(f"=> Model: {modname}")
        network = model_directory[modname]
        stitch_model = network()
        stitch_model.create_model(height=256, width=256, load_weights=False)
        plot_model(
            stitch_model.model,
            to_file=f"architectures/model_img/{modname}.png",
            show_shapes=True,
            show_layer_names=True,
        )


if __name__ == "__main__":
    """
    Plot the models
    """
    if args.supervised:
        with tf.device("/GPU:0"):
            train(
                height=cfg.patch_size,
                width=cfg.patch_size,
                save_arch=args.save_model_img,
                nb_epochs=args.nb_epochs,
                batch_size=args.batch_size,
                load_weights=args.load_weights,
            )
    else:
        with tf.device("/GPU:0"):
            un_train(
                height=cfg.un_patch_size,
                width=cfg.un_patch_size,
                save_arch=args.save_model_img,
                nb_epochs=args.nb_epochs,
                batch_size=args.batch_size,
                load_weights=args.load_weights,
            )
    # save_model_plots()
