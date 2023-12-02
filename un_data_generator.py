import os.path

import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

import prepare_data as psd
import constant as cfg
import gc


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(
        self,
        list_ids,
        config_data,
        callee=None,
        batch_size=32,
        dim=(256, 256),
        n_channels=15,
        shuffle=True,
    ):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.list_ids = list_ids
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.callee = callee
        self.config_data = config_data  # psd.read_json_file(psd.config_file)
        self.dataset_index_range = []
        self.total_patch_per_img = []
        end_range = 0
        for idx in range(self.config_data["total_dataset"]):
            nb_patch_per_img = (
                self.config_data[str(idx)]["patchX"]
                * self.config_data[str(idx)]["patchY"]
            )
            end_range += self.config_data[str(idx)]["nb_imgs"] * nb_patch_per_img

            self.total_patch_per_img.append(nb_patch_per_img)
            self.dataset_index_range.append(end_range)

        self.indexes = np.arange(len(self.list_ids))
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_ids_temp)

        return X, y

    # @profile
    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        # self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            self.shuffle = False
        gc.collect()
        keras.backend.clear_session()

    def __get_dataset_patch(self, index):
        # Find scene id
        dataset_id = -1
        adj_idx = index

        for sc_id, end_r in enumerate(self.dataset_index_range):
            if index < end_r:
                dataset_id = sc_id
                if sc_id >= 1:
                    adj_idx = index - self.dataset_index_range[sc_id - 1]
                break

        if dataset_id < 0:
            raise ValueError("Invalid index %d" % index)

        # print("Adjusted index", adj_idx)
        total_imgs = self.config_data[str(dataset_id)]["nb_imgs"]
        nb_cameras = self.config_data[str(dataset_id)]["nb_cameras"]

        img_idx, patch_id = divmod(adj_idx, self.total_patch_per_img[dataset_id])
        patchx_id, patchy_id = divmod(
            patch_id, self.config_data[str(dataset_id)]["patchY"]
        )

        # img_idx, patch_id = divmod(adj_idx, self.config_data[str(dataset_id)]["patchX"])
        # patchx_id, patchy_id = divmod(patch_id, self.config_data[str(dataset_id)]["patchY"])

        return dataset_id, patchx_id, patchy_id, img_idx, total_imgs, nb_cameras

    def __data_generation(self, list_ids_temp):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # y = np.empty((self.batch_size, *self.dim, 3))
        print("[%s]: Len of list_ids_temp: %d" % (self.callee, len(list_ids_temp)))
        X = np.zeros((self.batch_size, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size, *self.dim, 3))

        # Generate data
        for i, ID in enumerate(list_ids_temp):
            # Store sample
            (
                dataset_id,
                patchx_id,
                patchy_id,
                img_idx,
                total_imgs,
                nb_cameras,
            ) = self.__get_dataset_patch(ID)
            ts = psd.TrainingSample(
                datasetID=dataset_id,
                imgID=img_idx,
                patchX=patchx_id,
                patchY=patchy_id,
                image_folder=cfg.un_image_folder,
                nb_cameras=nb_cameras,
            )

            if os.path.exists(ts.get_sample_path()):
                print(
                    f"==> index: {ID}, {ts.get_sample_path()}, check: {os.path.exists(ts.get_sample_path())}"
                )
            X0 = ts.load_sample()
            y0 = ts.load_target()

            X[
                i,
            ] = X0
            y[
                i,
            ] = y0

        return X, y

    def __img_path_generation(self, list_ids):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization

        X = []

        # Generate data
        for i, ID in enumerate(list_ids):
            # Store sample
            (
                dataset_id,
                patchx_id,
                patchy_id,
                img_id,
                total_imgs,
                nb_cameras,
            ) = self.__get_dataset_patch(ID)
            X.append((dataset_id, patchx_id, patchy_id, img_id, total_imgs, nb_cameras))

        return X

    def test_path_generator(self, index_list):
        return self.__img_path_generation(index_list)

    def generate_img_path(self):
        return self.__img_path_generation(self.list_ids)


## Loading function for self-supervised training
def un_load_data_sample(img_paths, dim, n_channels, training_folder):

    dataset_id, patchx_id, patchy_id, img_id, total_imgs, nb_cameras = img_paths

    ts = psd.TrainingSample(
        datasetID=dataset_id,
        imgID=img_id,
        patchX=patchx_id,
        patchY=patchy_id,
        image_folder=training_folder.decode("utf-8"),
        nb_cameras=nb_cameras,
    )
    X = ts.load_sample()
    y = ts.load_target()
    # print("===================================", X.shape)

    return X, y


## Loading function for self-supervised training
# @tf.function
def un_read_img_dataset(
    list_ids,
    config_data,
    callee,
    batch_size=32,
    dim=(256, 256),
    n_channels=15,
    shuffle=True,
    seed=None,
    buffer_size=tf.data.AUTOTUNE,
):

    print("callee: %s" % callee, len(list_ids))
    data_gen = DataGenerator(list_ids, config_data, dim=dim)
    list_paths = data_gen.generate_img_path()

    dataset = tf.data.Dataset.from_tensor_slices(list_paths)

    dataset = dataset.map(
        lambda item1, item2=dim, item3=n_channels, item4=str(
            cfg.un_image_folder
        ): tf.numpy_function(
            un_load_data_sample, [item1, item2, item3, item4], [tf.float32, tf.float32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )  #

    return dataset.repeat(1).batch(batch_size=batch_size).prefetch(buffer_size)


if __name__ == "__main__":

    config_data = psd.read_json_file(cfg.un_config_img_output)

    d1 = DataGenerator(
        [*range(config_data["total_samples"])],
        config_data,
        dim=(cfg.un_patch_size, cfg.un_patch_size),
    )

    list_paths = d1.generate_img_path()
    print("------------------------------------------")
    print(list_paths[0:2])
    X, Y = un_load_data_sample(
        list_paths[2], dim=None, n_channels=None, training_folder=str(cfg.un_image_folder)
    )
    print(X.shape, " - ", Y.shape)
