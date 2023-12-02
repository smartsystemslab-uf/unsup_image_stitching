from __future__ import division, print_function

import json
import os
import random

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

import constant as cfg
import img_utils

# sys.path.append(cfg.pano_libs_dir)
# import pylab as plt
import panowrapper as pw


def dict_raise_on_duplicates(ordered_pairs):
    """Reject duplicate keys."""
    d = {}
    for k, v in ordered_pairs:
        if k in d:
            raise ValueError("duplicate key: %r" % (k,))
        else:
            d[k] = v
    return d


def write_json_file(filename, data, mode="w+"):
    if mode == "w+":
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode) as fp:
        json.dump(data, fp, indent=4)


def read_json_file(filename):
    try:
        with open(filename, "r") as fp:
            mdata = dict(json.load(fp, object_pairs_hook=dict_raise_on_duplicates))
    except IOError:
        print(f"File not found: {filename}.")
        mdata = dict()

    return mdata


def read_json_arr(filename):
    try:
        with open(filename, "r") as fp:
            mdata = json.loads(fp.read())
    except IOError:
        print(f"File not found: {filename}.")

    return mdata


def remove_file(filename):
    os.remove(filename) if os.path.exists(filename) else None


class TrainingSample:
    def __init__(self, datasetID, imgID, patchX, patchY, image_folder, nb_cameras=None):
        # self.sample_id = sampleID
        self.datasetID = datasetID
        self.img_id = imgID
        self.patch_x = patchX
        self.patch_y = patchY
        self.nb_cameras = nb_cameras

        self.dataset_path = f"{image_folder}/{datasetID}"
        self.sample_folder = f"{self.dataset_path}/X"
        self.target_folder = f"{self.dataset_path}/y"

    def get_sample_path(self):
        self.sample_path = f"{self.sample_folder}/imgID{self.img_id}_patchID{self.patch_x}_{self.patch_y}"
        return f"{self.sample_path}.npz"

    def get_target_path(self):
        # if camera_id:
        #     if not 0 < camera_id < self.nb_cameras:
        #         cfg.PRINT_WARNING(f"Camera ID is out of range: camera_id: {camera_id}, self.nb_cameras: {self.nb_cameras}")
        #     # Unsupervised target path
        #     self.target_path = f"{self.target_folder}/imgID{self.img_id}_camID{camera_id}_patchID{self.patch_x}_{self.patch_y}"
        # else:
        #     self.target_path = f"{self.target_folder}/imgID{self.img_id}_patchID{self.patch_x}_{self.patch_y}"
        self.target_path = f"{self.target_folder}/imgID{self.img_id}_patchID{self.patch_x}_{self.patch_y}"
        return f"{self.target_path}.npz"

    def save_sample(self, data, shuffle=True):
        if shuffle:
            # Total number of channels
            c = data.shape[-1]
            # Number of layers
            n = np.abs(cfg.sandfall_layer)
            # Shuffle layer order on the training samples
            data[:, :, np.arange(c)] = data[
                :, :, np.random.permutation(np.arange(c).reshape(n, 3)).ravel()
            ]
        np.savez_compressed(self.get_sample_path(), data=data)

    def save_target(self, data, shuffle=True):
        if shuffle:
            # Total number of channels
            c = data.shape[-1]
            # Number of images
            n = int(c // 3)
            # Shuffle layer order on the training samples
            data[:, :, np.arange(c)] = data[
                :, :, np.random.permutation(np.arange(c).reshape(n, 3)).ravel()
            ]
        np.savez_compressed(self.get_target_path(), data=data)

    def load_sample(self):
        return np.load(self.get_sample_path())["data"]

    def load_target(self):
        return np.load(self.get_target_path())["data"]


class Dataset:
    def __init__(
        self,
        datasetID,
        total_img,
        nb_cameras,
        nb_img_generate,
        img_pattern,
        image_folder,
        imgfs="MCMI",
        input_img_dir=None,
        scale_factor=1.0,
        config_output_file=None,
        training_config_dict=None,
    ):
        """
        Class to Generate the dataset from Pano Stitcher
        :param datasetID:
        :param size_x: patch_x size
        :param size_y: patch_y size
        :param nb_cameras: total number of cameras for this dataset
        :param scale_factor: Scale factor to resize the output images
        :param nb_img_generate: total number of image to generate
        :param total_img: total number of images for this dataset
        :param img_pattern: image pattern to retrieve image files on disk. This pattern is
            a python f-string and should contain the camera id and the image id fields
            Example: Terrace/Input/{camID:05d}/{imgID:05d}.jpg
        :param image_folder: Training folder to store images
        :param input_img_dir: The input image directory if provided
        :param imgfs: Image File Structure (DFS)
                        "MCMI: Multi-Camera Multi-Image (indicate both the 'camID' and the 'imgID' in the file pattern \n"
                        "SCMI: Single-Camera Multi-Image ('imgID' in the file pattern"
                        "MCSI: Multi-Camera Multi-Image ('camID' in the file pattern"
                        "LIST: list of files image to stitch and provide the list in the --files param"
                        "IDIR: Image Directory of files with *.jpg, *.jpeg, *.png, and *.bmp will be retrieve in the folder indicate by --imgdir"
        :param training_config_dict: Default: None. Dictionary of the data setting for this dataset
        """
        self.dataset_id = datasetID
        self.total_img = total_img
        self.imgfs = imgfs
        self.img_pattern = img_pattern
        self.input_img_dir = input_img_dir
        self.nb_cameras = nb_cameras
        self.scale_factor = scale_factor
        self.nb_img_generate = nb_img_generate
        self.image_folder = f"{image_folder}"
        self.config_output_file = f"{config_output_file}"
        self.dataset_path = f"{image_folder}/{datasetID}"
        self.target_img_path = f"{self.dataset_path}/target"
        self.sample_folder = f"{self.dataset_path}/X"
        self.target_path = f"{self.dataset_path}/y"
        # self.__is_supervised = True # By default the data are generated for supervised training

        self.dataset_settings = training_config_dict
        if "total_dataset" not in self.dataset_settings:
            self.dataset_settings.clear()
            self.dataset_settings["total_dataset"] = 0
            self.dataset_settings["total_samples"] = 0
            write_json_file(self.config_output_file, data=self.dataset_settings)

        if str(self.dataset_id) not in self.dataset_settings:
            self.dataset_settings[str(self.dataset_id)] = {
                "nb_imgs": 0,
                "nb_cameras": self.nb_cameras,
                "scale_factor": self.scale_factor,
                "patchX": 0,
                "patchY": 0,
                "patchSizeX": 0,
                "patchSizeY": 0,
            }
            self.dataset_settings["total_dataset"] += 1

        os.makedirs(self.dataset_path, exist_ok=True)
        os.makedirs(self.sample_folder, exist_ok=True)
        os.makedirs(self.target_path, exist_ok=True)
        os.makedirs(self.target_img_path, exist_ok=True)

    def __convert_mat_np(self, mat):
        p = np.array(mat, copy=False)
        p[p < 0] = 0  # Replace negative values with zeros
        p[p > 1] = 1  # clip values greater than ones
        p[~np.isfinite(p)] = 0  # replace non-finite values (nan, inf, etc.) with zeros
        return p

    def set_dataset_settings(self, key, value):
        self.dataset_settings[str(self.dataset_id)][key] = value

    def get_dataset_settings(self, key, default_value=None):
        return self.dataset_settings[str(self.dataset_id)].get(key, default_value)

    def increment_settings(self, key, value):
        if key in self.dataset_settings[str(self.dataset_id)]:
            self.dataset_settings[str(self.dataset_id)][key] += value
        else:
            self.dataset_settings[str(self.dataset_id)][key] = value

    def generate_dataset(self):
        """Generate dataset for the supervised training"""

        panow = pw.PanoWrapper(
            scale_factor_x=self.scale_factor, scale_factor_y=self.scale_factor
        )

        if self.imgfs == "MCMI":
            # Trying to find the optimal projection matrix
            # by initializing the pano stitcher object
            for img_id in range(self.total_img):

                file_list = [
                    self.img_pattern.format(camID=i, imgID=img_id)
                    for i in range(self.nb_cameras)
                ]

                print(file_list)
                try:
                    panow.init_pano_stitcher(
                        file_list, multi_band_blend=cfg.sandfall_layer
                    )
                    break
                except:
                    print(f"Error: Cannot stitch image [{img_id}]")

            if not panow.is_pano_initialize():
                raise RuntimeError(
                    "Failed to find the projection parameters. Please add calibration "
                    "images in the same director as the images"
                )

            sample_img_id = random.sample(range(self.total_img), self.nb_img_generate)

            for img_id in sample_img_id:
                file_list = [
                    self.img_pattern.format(camID=i, imgID=img_id)
                    for i in range(self.nb_cameras)
                ]

                print(file_list)
                mat_merge = panow.build_pano(
                    file_list, multi_band_blend=cfg.sandfall_layer
                )
                pmat_merge = self.__convert_mat_np(mat_merge)
                mat_target = panow.build_pano(file_list, multi_band_blend=20)
                pmat_target = self.__convert_mat_np(mat_target)
                panow.write_img(
                    f'{self.target_img_path}/tgID{self.get_dataset_settings("nb_imgs")}.jpg',
                    mat_target,
                )

                self.write_sample(
                    self.get_dataset_settings("nb_imgs"),
                    pmat_merge,
                    pmat_target,
                    cfg.patch_step,
                    cfg.patch_size,
                )

        else:
            files = []
            if self.imgfs == "SCMI":
                files = [
                    self.img_pattern.format(imgID=img_id)
                    for img_id in range(self.nb_img_generate)
                ]
            elif self.imgfs == "MCSI":
                files = [self.img_pattern.format(camID=i) for i in range(self.nb_cameras)]
            elif self.imgfs == "IDIR":
                exts = ["jpg", "jpeg", "png", "bmp"]
                for ext in exts:
                    import re

                    files.extend(
                        [
                            os.path.join(self.input_img_dir, filename)
                            for filename in os.listdir(self.input_img_dir)
                            if re.search(r"\." + ext + "$", filename, re.IGNORECASE)
                        ]
                    )
                    # files.extend(glob.glob(os.path.join(self.input_img_dir, ext)))
            else:
                ValueError("Please indicate the dataset file structure for your images")
            self.nb_cameras = len(files)
            # self.dataset_settings["nb_cameras"] = self.nb_cameras
            self.set_dataset_settings("nb_cameras", self.nb_cameras)

            try:
                panow.init_pano_stitcher(files, multi_band_blend=cfg.sandfall_layer)
            except:
                print(f"Error: Cannot stitch image [{files}]")

            if not panow.is_pano_initialize():
                raise RuntimeError(
                    "Failed to find the projection parameters. Please add calibration "
                    "images in the same director as the images"
                )

            print(files)
            mat_merge = panow.build_pano(files, multi_band_blend=cfg.sandfall_layer)
            pmat_merge = self.__convert_mat_np(mat_merge)
            mat_target = panow.build_pano(files, multi_band_blend=20)
            pmat_target = self.__convert_mat_np(mat_target)
            panow.write_img(
                f'{self.target_img_path}/tgID{self.get_dataset_settings("nb_imgs")}.jpg',
                mat_target,
            )

            self.write_sample(
                self.get_dataset_settings("nb_imgs"),
                pmat_merge,
                pmat_target,
                cfg.patch_step,
                cfg.patch_size,
            )
        return panow.img_height, panow.img_width, panow.img_channels

    def generate_un_dataset(self):
        """Generate Dataset for Unsupervised Training"""

        panow = pw.PanoWrapper(
            scale_factor_x=self.scale_factor, scale_factor_y=self.scale_factor
        )

        if self.imgfs == "MCMI":
            # Trying to find the optimal projection matrix
            # by initializing the pano stitcher object
            for img_id in range(self.total_img):

                file_list = [
                    self.img_pattern.format(camID=i, imgID=img_id)
                    for i in range(self.nb_cameras)
                ]

                print(file_list)
                try:
                    panow.init_pano_stitcher(
                        file_list, multi_band_blend=cfg.sandfall_layer
                    )
                    break
                except:
                    print(f"Error: Cannot stitch image [{img_id}]")

            if not panow.is_pano_initialize():
                raise RuntimeError(
                    "Failed to find the projection parameters. Please add calibration "
                    "images in the same director as the images"
                )

            sample_img_id = random.sample(range(self.total_img), self.nb_img_generate)

            for img_id in sample_img_id:
                file_list = [
                    self.img_pattern.format(camID=i, imgID=img_id)
                    for i in range(self.nb_cameras)
                ]

                print("File list: ", file_list)
                mat_merge = panow.build_pano(
                    file_list, multi_band_blend=cfg.sandfall_layer
                )
                pmat_merge = self.__convert_mat_np(mat_merge)

                mat_multiband = panow.build_pano(file_list, multi_band_blend=20)
                panow.write_img(
                    f'{self.target_img_path}/tgID{self.get_dataset_settings("nb_imgs")}.jpg',
                    mat_multiband,
                )

                mat_target = panow.build_pano(file_list, multi_band_blend=-1)
                pmat_target = self.__convert_mat_np(mat_target)

                self.write_sample(
                    self.get_dataset_settings("nb_imgs"),
                    pmat_merge,
                    pmat_target,
                    cfg.un_patch_step,
                    cfg.un_patch_size,
                )

        else:
            files = []
            if self.imgfs == "SCMI":
                files = [
                    self.img_pattern.format(imgID=img_id)
                    for img_id in range(self.nb_img_generate)
                ]
            elif self.imgfs == "MCSI":
                files = [self.img_pattern.format(camID=i) for i in range(self.nb_cameras)]
            elif self.imgfs == "IDIR":
                exts = ["jpg", "jpeg", "png", "bmp"]
                for ext in exts:
                    import re

                    files.extend(
                        [
                            os.path.join(self.input_img_dir, filename)
                            for filename in os.listdir(self.input_img_dir)
                            if re.search(r"\." + ext + "$", filename, re.IGNORECASE)
                        ]
                    )
                    # files.extend(glob.glob(os.path.join(self.input_img_dir, ext)))
            else:
                ValueError("Please indicate the dataset file structure for your images")
            self.nb_cameras = len(files)
            # self.dataset_settings["nb_cameras"] = self.nb_cameras
            self.set_dataset_settings("nb_cameras", self.nb_cameras)

            try:
                panow.init_pano_stitcher(files, multi_band_blend=cfg.sandfall_layer)
            except:
                print(f"Error: Cannot stitch image [{files}]")

            if not panow.is_pano_initialize():
                raise RuntimeError(
                    "Failed to find the projection parameters. Please add calibration "
                    "images in the same director as the images"
                )

            print(files)
            mat_merge = panow.build_pano(files, multi_band_blend=cfg.sandfall_layer)
            pmat_merge = self.__convert_mat_np(mat_merge)

            mat_multiband = panow.build_pano(files, multi_band_blend=20)
            panow.write_img(
                f'{self.target_img_path}/tgID{self.get_dataset_settings("nb_imgs")}.jpg',
                mat_multiband,
            )

            mat_target = panow.build_pano(files, multi_band_blend=-1)
            pmat_target = self.__convert_mat_np(mat_target)

            self.write_sample(
                self.get_dataset_settings("nb_imgs"),
                pmat_merge,
                pmat_target,
                cfg.un_patch_step,
                cfg.un_patch_size,
            )
        return panow.img_height, panow.img_width, panow.img_channels

    def write_sample(self, img_id, img, target, step, patch_size):

        img_patches = img_utils.make_raw_patches(
            img, step=step, patch_size=patch_size, channels=img.shape[-1], verbose=1
        )
        target_patches = img_utils.make_raw_patches(
            target, step=step, patch_size=patch_size, channels=target.shape[-1], verbose=1
        )

        print("Patches shape: ", img_patches.shape, ", target.shape: ", target.shape)
        self.set_dataset_settings("patchX", img_patches.shape[0])
        self.set_dataset_settings("patchY", img_patches.shape[1])
        self.set_dataset_settings("patchSizeX", patch_size)
        self.set_dataset_settings("patchSizeY", patch_size)
        self.increment_settings("nb_imgs", 1)
        self.dataset_settings["total_samples"] += (
            img_patches.shape[0] * img_patches.shape[1]
        )

        def __save_patch(patchIdx, patchIdy, pbar):
            img_patch = img_patches[patchIdx, patchIdy, 0, :, :]
            target_patch = target_patches[patchIdx, patchIdy, 0, :, :]

            train_sample_obj = TrainingSample(
                datasetID=self.dataset_id,
                imgID=img_id,
                patchX=patchIdx,
                patchY=patchIdy,
                image_folder=self.image_folder,
                nb_cameras=self.nb_cameras,
            )

            train_sample_obj.save_sample(img_patch)
            train_sample_obj.save_target(target_patch)
            pbar.update(1)

        with tqdm(total=img_patches.shape[0] * img_patches.shape[1]) as pbar:
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:

                for patchIdx in range(img_patches.shape[0]):
                    for patchIdy in range(img_patches.shape[1]):
                        ex.submit(__save_patch, patchIdx, patchIdy, pbar)

        # Update config file each iteration
        write_json_file(self.config_output_file, data=self.dataset_settings)
        return


def prepare_data_live(args):
    """Prepare data set for image stitching"""

    print("Argument: ", args, " - is supervised? ", args.supervised)
    image_folder = cfg.image_folder if args.supervised else cfg.un_image_folder
    config_output_file = (
        cfg.config_img_output if args.supervised else cfg.un_config_img_output
    )
    patch_size = cfg.patch_size if args.supervised else cfg.un_patch_size
    patch_step = cfg.patch_step if args.supervised else cfg.un_patch_step

    # Read the input of the new data file settings
    data_file_settings = read_json_file(cfg.config_img_input)

    # Read the output config file of all the training data already generated.
    training_config_dict = read_json_file(config_output_file)

    print(data_file_settings)
    global_scale_factor = data_file_settings.get("global_scale_factor", 1.0)
    m_dataset_id = 0
    for datasetID, dataset_dc in data_file_settings.items():
        print("datasetID", datasetID, ", dataset_dc", dataset_dc)
        if not isinstance(dataset_dc, dict):
            continue

        total_img = dataset_dc.get("total_img", 0)
        nb_cameras = dataset_dc.get("nb_cameras", 0)
        nb_img_generate = dataset_dc.get("nb_img_generate", 0)
        img_pattern = dataset_dc.get("img_pattern", None)
        input_img_dir = dataset_dc.get("input_img_dir", None)
        imgfs = dataset_dc.get("IMGFS", "MCMI")
        scale_factor = dataset_dc.get("scale_factor", 1.0)
        # print("==>", img_pattern)
        # scale_factor = scale_factor*2 if args.supervised else scale_factor*2
        scale_factor = scale_factor * global_scale_factor
        ds = Dataset(
            f"{m_dataset_id}",
            total_img,
            nb_cameras,
            nb_img_generate,
            img_pattern,
            image_folder,
            imgfs,
            input_img_dir,
            scale_factor,
            config_output_file,
            training_config_dict,
        )
        m_dataset_id += 1

        if args.supervised:
            img_h, img_w, img_c = ds.generate_dataset()
        else:
            img_h, img_w, img_c = ds.generate_un_dataset()
        print(f"Image size: {img_h}x{img_w}x{img_c}")

        if img_h and img_w:
            # Compute the number of patches windows
            nb_windows_h = int((img_h - patch_size) / patch_step)
            # nb_windows_w = int((img_w - path_size) / patch_step)

            # Generate a maximum of 5 level including 1
            a = np.linspace(nb_windows_h, 0, num=5, dtype=int)
            # ensure uniqueness and ignore the first one.
            c = a[np.sort(np.unique(a, return_index=True)[1])][1:]
            list_kh = c[c >= 0]  # remove negative number
            print(f"list_kh => {list_kh}")
            for k_h in list_kh:
                exp_out_size_h = patch_step * k_h + patch_size
                scale_factor_h = img_h / exp_out_size_h

                scale_factor_h *= scale_factor
                m_ds = Dataset(
                    f"{m_dataset_id}",
                    total_img,
                    nb_cameras,
                    nb_img_generate,
                    img_pattern,
                    image_folder,
                    imgfs,
                    input_img_dir,
                    scale_factor_h,
                    config_output_file,
                    training_config_dict,
                )
                m_dataset_id += 1

                if args.supervised:
                    m_ds.generate_dataset()
                else:
                    m_ds.generate_un_dataset()
                print(
                    f"k_h: {k_h} - Image size [scale={scale_factor_h}]: {img_h}x{img_w}x{img_c}"
                )
                # for k_w in range(nb_windows_w-1, -1, -1):
                #     exp_out_size_w = patch_step * k_w + patch_size
                #     scale_factor_w = img_w / exp_out_size_w


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--supervised",
        default=True,
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
    )

    args = parser.parse_args()

    prepare_data_live(args)
