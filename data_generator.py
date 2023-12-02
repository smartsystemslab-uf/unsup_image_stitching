import gc
import os.path

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import constant as cfg
import prepare_data as psd


# class DataGenerator(keras.utils.Sequence):
#     """Generates data for Keras"""

#     def __init__(self, list_ids, config_data, callee=None, batch_size=32, dim=(256, 256), n_channels=15, shuffle=True):
#         """Initialization"""
#         self.dim = dim
#         self.batch_size = batch_size
#         self.list_ids = list_ids
#         self.n_channels = n_channels
#         self.shuffle = shuffle
#         self.callee = callee
#         self.config_data = config_data  # psd.read_json_file(psd.config_file)
#         self.dataset_index_range = []
#         self.total_patch_per_img = []
#         end_range = 0
#         for idx in range(self.config_data["total_dataset"]):
#             nb_patch_per_img = self.config_data[str(idx)]["patchX"] * \
#                 self.config_data[str(idx)]["patchY"]
#             end_range += self.config_data[str(idx)]["nb_imgs"] * nb_patch_per_img

#             self.total_patch_per_img.append(nb_patch_per_img)
#             self.dataset_index_range.append(end_range)

#         self.indexes = np.arange(len(self.list_ids))
#         self.on_epoch_end()

#     def __len__(self):
#         """Denotes the number of batches per epoch"""
#         return int(np.floor(len(self.list_ids) / self.batch_size))

#     def __getitem__(self, index):
#         """Generate one batch of data"""
#         # Generate indexes of the batch
#         indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

#         # Find list of IDs
#         list_ids_temp = [self.list_ids[k] for k in indexes]

#         # Generate data
#         X, y = self.__data_generation(list_ids_temp)

#         return X, y

#     # @profile
#     def on_epoch_end(self):
#         """Updates indexes after each epoch"""
#         # self.indexes = np.arange(len(self.list_ids))
#         if self.shuffle:
#             np.random.shuffle(self.indexes)
#             self.shuffle = False
#         gc.collect()
#         keras.backend.clear_session()

#     def __get_dataset_patch(self, index):
#         # Find scene id
#         dataset_id = -1
#         adj_idx = index

#         for sc_id, end_r in enumerate(self.dataset_index_range):
#             if index < end_r:
#                 dataset_id = sc_id
#                 if sc_id >= 1:
#                     adj_idx = index - self.dataset_index_range[sc_id - 1]
#                 break

#         if dataset_id < 0:
#             raise ValueError(f"Invalid index: {index}")

#         # print("Adjusted index", adj_idx)
#         total_imgs = self.config_data[str(dataset_id)]["nb_imgs"]
#         nb_cameras = self.config_data[str(dataset_id)]["nb_cameras"]

#         img_idx, patch_id = divmod(adj_idx, self.total_patch_per_img[dataset_id])
#         patchx_id, patchy_id = divmod(patch_id, self.config_data[str(dataset_id)]["patchY"])

#         # img_idx, patch_id = divmod(adj_idx, self.config_data[str(dataset_id)]["patchX"])
#         # patchx_id, patchy_id = divmod(patch_id, self.config_data[str(dataset_id)]["patchY"])

#         return dataset_id, patchx_id, patchy_id, img_idx, total_imgs, nb_cameras

#     def __data_generation(self, list_ids_temp):
#         """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
#         # Initialization
#         # X = np.empty((self.batch_size, *self.dim, self.n_channels))
#         # y = np.empty((self.batch_size, *self.dim, 3))
#         print("[%s]: Len of list_ids_temp: %d" % (self.callee, len(list_ids_temp)))
#         X = np.zeros((self.batch_size, *self.dim, self.n_channels))
#         y = np.zeros((self.batch_size, *self.dim, 3))

#         # Generate data
#         for i, ID in enumerate(list_ids_temp):
#             # Store sample
#             dataset_id, patchx_id, patchy_id, img_idx, total_imgs, nb_cameras = self.__get_dataset_patch(ID)
#             ts = psd.TrainingSample(datasetID=dataset_id, imgID=img_idx, patchX=patchx_id,
#                                     patchY=patchy_id, image_folder=cfg.image_folder, nb_cameras=nb_cameras)

#             if os.path.exists(ts.get_sample_path()):
#                 print(f"==> index: {ID}, {ts.get_sample_path()}, check: {os.path.exists(ts.get_sample_path())}")
#             X0 = ts.load_sample()
#             y0 = ts.load_target()

#             X[i, ] = X0
#             y[i, ] = y0

#         return X, y

#     def __img_path_generation(self, list_ids):
#         """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
#         # Initialization

#         X = []

#         # Generate data
#         for i, ID in enumerate(list_ids):
#             # Store sample
#             dataset_id, patchx_id, patchy_id, img_id, total_imgs, nb_cameras = self.__get_dataset_patch(ID)
#             X.append((dataset_id, patchx_id, patchy_id, img_id, total_imgs, nb_cameras))

#         return X

#     def test_get_scene_patch(self, index):
#         return self.__get_dataset_patch(index)

#     def test_data_generator(self, index_list):
#         # Store sample
#         paths = []
#         for ID in index_list:
#             dataset_id, patchx_id, patchy_id, img_idx, total_imgs, nb_cameras = self.__get_dataset_patch(ID)
#             ts = psd.TrainingSample(datasetID=dataset_id, imgID=img_idx, patchX=patchx_id,
#                                     patchY=patchy_id, image_folder=cfg.image_folder, nb_cameras=nb_cameras)

#             paths.append(ts.get_sample_path())
#             if not os.path.exists(ts.get_sample_path()):
#                 print(f"==> index: {ID}, {ts.get_sample_path()}, check: {os.path.exists(ts.get_sample_path())}")

#         print("==> Checking for duplicate")
#         if len(paths) != len(set(paths)):
#             print("There are duplicates.")
#         else:
#             print("All file names are unique")

#         print("test_data_generator Done!")
#         return

#     def test_path_generator(self, index_list):
#         return self.__img_path_generation(index_list)

#     def generate_img_path(self):
#         return self.__img_path_generation(self.list_ids)


def new_py_function(func, inp, Tout, name=None):
    def wrapped_func(*flat_inp):
        reconstructed_inp = tf.nest.pack_sequence_as(
            inp, flat_inp, expand_composites=True
        )
        out = func(*reconstructed_inp)
        return tf.nest.flatten(out, expand_composites=True)

    flat_Tout = tf.nest.flatten(Tout, expand_composites=True)
    flat_out = tf.py_function(
        func=wrapped_func,
        inp=tf.nest.flatten(inp, expand_composites=True),
        Tout=[_tensor_spec_to_dtype(v) for v in flat_Tout],
        name=name,
    )
    spec_out = tf.nest.map_structure(_dtype_to_tensor_spec, Tout, expand_composites=True)
    out = tf.nest.pack_sequence_as(spec_out, flat_out, expand_composites=True)
    return out


def _dtype_to_tensor_spec(v):
    return tf.TensorSpec(None, v) if isinstance(v, tf.dtypes.DType) else v


def _tensor_spec_to_dtype(v):
    return v.dtype if isinstance(v, tf.TensorSpec) else v


# def load_data_sample(img_paths, dim, n_channels, training_folder):

#     dataset_id, patchx_id, patchy_id, img_id, total_imgs, nb_cameras = img_paths

#     ts = psd.TrainingSample(datasetID=dataset_id, imgID=img_id, patchX=patchx_id,
#                             patchY=patchy_id, image_folder=training_folder.decode("utf-8"), nb_cameras=nb_cameras)
#     X = ts.load_sample()
#     y = ts.load_target()

#     return X, y

# # @tf.function
# def load_data_sample(sample_id, config_data, training_folder):
#     if type(sample_id) != int:
#         sample_id = int(sample_id.numpy())
#         assert type(sample_id) == int

#     dataset_id, patchx_id, patchy_id, img_id, total_imgs, nb_cameras = get_sample_from_index(sample_id, config_data)

#     ts = psd.TrainingSample(datasetID=dataset_id, imgID=img_id, patchX=patchx_id,
#                             patchY=patchy_id, image_folder=training_folder.numpy().decode("utf-8"), nb_cameras=nb_cameras)
#     X = ts.load_sample()
#     y = ts.load_target()

#     return X, y

# @tf.function
def load_data_sample(sample_path, target_path):

    with np.load(sample_path.numpy().decode("utf-8")) as data:
        X = data["data"]

    with np.load(target_path.numpy().decode("utf-8")) as data:
        y = data["data"]

    return X, y


def get_path_from_index(index, config_data, training_folder):

    if type(index) != int:
        index = int(index.numpy())
        assert type(index) == int

    dataset_id = -1
    adj_idx = index
    end_range = 0

    for idx in range(config_data["total_dataset"]):
        nb_patch_per_img = (
            config_data[str(idx)]["patchX"] * config_data[str(idx)]["patchY"]
        )

        prev_end_range = end_range
        end_range += config_data[str(idx)]["nb_imgs"] * nb_patch_per_img

        if index < end_range:
            dataset_id = idx
            if idx > 0:
                adj_idx = index - prev_end_range
            break

    if dataset_id < 0:
        raise ValueError(f"Invalid index: {index}")

    nb_patch_per_img = (
        config_data[str(dataset_id)]["patchX"] * config_data[str(dataset_id)]["patchY"]
    )

    total_imgs = config_data[str(dataset_id)]["nb_imgs"]
    nb_cameras = config_data[str(dataset_id)]["nb_cameras"]

    img_idx, patch_id = np.divmod(adj_idx, nb_patch_per_img)
    patchx_id, patchy_id = np.divmod(patch_id, config_data[str(dataset_id)]["patchY"])

    ts = psd.TrainingSample(
        datasetID=dataset_id,
        imgID=img_idx,
        patchX=patchx_id,
        patchY=patchy_id,
        image_folder=training_folder.numpy().decode("utf-8"),
        nb_cameras=nb_cameras,
    )

    return ts.get_sample_path(), ts.get_target_path()


@tf.function
def read_img_dataset(
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

    dataset = tf.data.Dataset.from_tensor_slices(list_ids)

    dataset = dataset.map(
        lambda item1, item2=config_data, item3=str(cfg.image_folder): new_py_function(
            get_path_from_index, inp=[item1, item2, item3], Tout=[tf.string, tf.string]
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).cache()  #

    dataset = dataset.map(
        lambda sample_path, target_path: tf.py_function(
            load_data_sample,
            inp=[sample_path, target_path],
            Tout=[tf.float32, tf.float32],
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )  #

    dataset = dataset.repeat(1).prefetch(tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size)  # .cache()

    return dataset


if __name__ == "__main__":

    config_data = psd.read_json_file(cfg.config_img_output)

    # d = DataGenerator([], config_data, dim=(cfg.patch_size, cfg.patch_size))
    # # for i in range(12387):
    # #     d.test_get_scene_patch(i)
    # # print(d.test_get_scene_patch(45 * 15))

    # # X, y = d.test_data_generator([0, 23, 54, 2050])
    # # print("x.shape: ", X.shape)
    # # print("y.shape: ", y.shape)

    # d.test_data_generator([*range(config_data["total_samples"])])

    print("==> For consistency in the total_sample")
    print(f'total_samples: {config_data["total_samples"]}')

    # print("Len X: ", len(X), X)

    # from sklearn.model_selection import train_test_split
    #
    # x = np.arange(12387)
    # x_train, x_test = train_test_split(x, test_size=0.20)
    # print(len(x_test))
    # print(len(x_train))

    # d1 = DataGenerator([*range(config_data["total_samples"])], config_data, dim=(cfg.patch_size, cfg.patch_size))

    # list_paths = d1.generate_img_path()
    # print(list_paths[0:2])
    # X, Y = load_data_sample(list_paths[2], dim=None, n_channels=None, training_folder=str(cfg.image_folder))
    # print(X.shape, " - ", Y.shape)

    # import random
    # for index in random.sample(range(config_data["total_samples"]), k=10):
    #     img_data_info = get_sample_from_index(index, config_data)
    #     print(f"index:{index} - {img_data_info}")
