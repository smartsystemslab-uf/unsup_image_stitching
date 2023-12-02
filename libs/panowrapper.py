from __future__ import print_function, division

import os
import sys
import time

import numpy as np
import random

import constant as cfg
import cv2

sys.path.append(cfg.pano_libs_dir)

try:
    import libpyopenpano as pano
except:
    ValueError(
        "Couldn't import 'libpyopenpano' library. You may need to use the shell "
        "script (*.sh files) to run this module or export LD_LIBRARY_PATH variable.\n"
        "    => Ex: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIB_DIR && python prepare_stitching_data.py"
    )


class PanoWrapper:
    def __init__(self, scale_factor_x=1.0, scale_factor_y=1.0, verbose=0):
        pano.init_config(cfg.pano_config_file)
        self.verbose = verbose
        self.scale_factor_x = scale_factor_x
        self.scale_factor_y = scale_factor_y
        self.pano_stitch = None
        self.pano_stitch_init = False
        self.img_height = 0
        self.img_width = 0
        self.img_channels = 0
        # help(pano)

        if self.verbose:
            pano.print_config()

    def print_config(self):
        pano.print_config()

    def is_pano_initialize(self):
        return self.pano_stitch_init

    def init_pano_stitcher(self, calib_files, multi_band_blend, shuffle=True):

        for img_path in calib_files:
            if not os.path.isfile(img_path):
                raise ValueError(f"Error: Image file {img_path} does not exists")

        # if shuffle:
        #     random.shuffle(calib_files)

        pano_stitch = pano.Stitcher(calib_files)
        self.pano_stitch_init = False
        self.pano_stitch = None
        try:
            mat = pano_stitch.build(
                self.scale_factor_x, self.scale_factor_y, multi_band_blend
            )
            self.img_height = mat.rows()
            self.img_width = mat.cols()
            self.img_channels = mat.channels()
            self.pano_stitch_init = True
            self.pano_stitch = pano_stitch
            return mat
        except:
            raise RuntimeError(
                f"Error: Failed to calibrate the input images {calib_files}"
            )

    def init_pano_from_dir(self, img_pattern, nb_cameras, nb_images):
        """
        Initialize the panorama stitcher object from a directory of camera images.
        :param img_pattern: The pattern string to retrieve images path and should contains the
                            camera id and the images id.
                            ex: DIR/Input/{camID:05d}/{imgID:05d}.jpg
        :param nb_cameras: total number of cameras.
        :param nb_images: Total number of images per camera that will be used to find the calibration parameters.
                            This could be less than the total number image available per camera on the disk.
        :return:
        """
        self.pano_stitch_init = False
        self.pano_stitch = None
        multi_band_blend = 0

        for img_id in range(0, nb_images):

            file_list = [
                img_pattern.format(camID=i, imgID=img_id) for i in range(nb_cameras)
            ]

            try:
                self.init_pano_stitcher(file_list, multi_band_blend)
                print(f"files: {file_list}")
                break
            except ValueError:
                cfg.PRINT_WARNING(
                    f"Error: Please check image exists to the specified location - {file_list}"
                )
                raise RuntimeError("Please, check filenames")
            except Exception as e:
                cfg.PRINT_WARNING(
                    f"{e}\nError: Cannot stitch images [{img_id}/{nb_images}]"
                )
                time.sleep(0.05)

        if not self.pano_stitch_init:
            raise RuntimeError(
                "Failed to find the projection parameters. Please add calibration "
                "images in the same director as the images"
            )

    def build_pano(self, img_paths, multi_band_blend, shuffle=True):
        if not self.pano_stitch:
            return None
        for img_path in img_paths:
            if not os.path.isfile(img_path):
                raise ValueError(f"Error: Image file {img_path} does not exists")
        # if shuffle:
        #     random.shuffle(img_paths)
        mat = self.pano_stitch.build_from_new_images(img_paths, multi_band_blend)
        return mat

    def write_img(self, path, mat):
        if not self.pano_stitch:
            return None
        pano.write_img(path, mat)

    def pano_stitch_single_camera(
        self,
        img_paths: list,
        out_filename=None,
        calib_files=None,
        return_img=False,
        multi_band_blend=0,
    ):
        """

        :param img_paths: The list of the image paths to stitched
        :param out_filename: The output filename
        :param calib_files: Camera calibration image file if exists
        :param multi_band_blend: Indicate the blending method
                                ** -1 for merge blending,
                                ** 0 for linear blending,
                                ** k>0 for multiband blending.
                                ** k<-1 and k>-20 trigger the execution of the sandfall method with the number of layers being abs(k).
                                Merge blending result cannot be saved directly a file.
                                ** k<-20 trigger seam blending
        :param return_img: if True the resulting stitched image will be returned.
                            The result image would be a numpy array
        :return: A numpy array of the stitched image
        """

        if calib_files:
            try:
                self.init_pano_stitcher(calib_files, multi_band_blend)
            except:
                cfg.PRINT_WARNING(
                    f"Error: Failed to calibrate the input images {calib_files}"
                )
                # if self.pano_stitch_init:
                #     self.init_pano_stitcher(img_paths, multi_band_blend)

            if not self.pano_stitch_init:
                try:
                    self.init_pano_stitcher(img_paths, multi_band_blend)
                except:
                    raise RuntimeError(
                        f"Error: Failed to calibrate the input images {calib_files}"
                    )

        if not self.pano_stitch_init:
            print("Please, initialize the pano object by provide the calibration files.")
            return None

        mat = self.build_pano(img_paths, multi_band_blend)

        if out_filename and not (multi_band_blend < 0 and multi_band_blend > -20):
            pano.write_img(out_filename, mat)

        if return_img:
            p = np.array(mat, copy=False)
            non_zeros_pixel = np.count_nonzero(p)
            p[p < 0] = 0  # Replace negative values with zeros
            p[p > 1] = 1  # Replace negative values with zeros
            h, w, c = p.shape
            X = np.zeros((1, h, w, c))

            X[0, :, :, :c] = p
            print(
                "********++++==> ",
                X.shape,
                " ~ ",
                p.dtype,
                ", non_zeros_pixels: ",
                non_zeros_pixel,
                ", multi_band_blend: ",
                multi_band_blend,
            )
            return X


if __name__ == "__main__":
    # prepare_data()

    import libpyopenpano as pano

    # help(pano)
    # Test Stitching
    pano.print_config()
    pano.init_config(cfg.pano_config_file)
    pano.print_config()

    mdata = [
        {
            "img_pattern": "/media/sf_Data/data_stitching/Terrace/Input/{:05d}/{:05d}.jpg",
            "out_dir": "/media/sf_Data/data_stitching/Terrace/Out2",
            "nb_cameras": 14,
            "total_img": 430,
        },
        {
            "img_pattern": "/media/sf_Data/data_stitching/Terrace/Input/{:05d}/{:05d}.jpg",
            "out_dir": "/media/sf_Data/data_stitching/Terrace/Out2",
            "nb_cameras": 14,
            "total_img": 430,
        },
    ]

    id = 0
    # img_dir = "/media/sf_Data/data_stitching/Terrace/Input/{:05d}/{:05d}.jpg"
    out_dir = mdata[id]["out_dir"]
    nb_camera = mdata[id]["nb_cameras"]
    total_img = mdata[id]["total_img"]
    os.makedirs(out_dir, exist_ok=True)
    output_result = None

    nb_stitched_img = 0
    stitcher = None
    for img_id in range(total_img):
        print(f"-----------------------------{img_id}------------------------")
        file_list = [
            mdata[id]["img_pattern"].format(i, img_id)
            for i in range(mdata[id]["nb_cameras"])
        ]
        output_result = f"{out_dir}/{img_id:05d}.jpg"

        print(file_list)
        print(output_result)

        stitcher = None
        try:
            stitcher = pano.Stitcher(file_list)
            mat = stitcher.build()
            pano.write_img(output_result, mat)
            break
        except:
            print(f"Error: Cannot stitch image [{img_id}] - [{output_result}]")

    print(f"First image stitched: {stitcher} --> Location: {output_result}")
    multi_band_blend = 0  # 0 is for linear blending
    time.sleep(10)
    for img_id in range(total_img):
        print(f"-----------------------------{img_id}------------------------")
        file_list = [
            mdata[id]["img_pattern"].format(i, img_id)
            for i in range(mdata[id]["nb_cameras"])
        ]
        output_result = f"{out_dir}/{img_id:05d}.jpg"

        try:
            print(f"Try to build from new images {img_id}/{total_img}")
            # print(file_list)
            mat = stitcher.build_from_new_images(file_list, multi_band_blend)
            print(f"Done building from new images {img_id}/{total_img}")
            # time.sleep(10)
            pano.write_img(output_result, mat)
        except:
            print(
                f"[build_from_new_images] Error: Cannot stitch image [{img_id}] - [{output_result}]"
            )

    # pano.test_extrema(mat, 1)
    print("done stitching!", nb_stitched_img)
    # pano.print_config()

    # p = np.array(mat, copy=False)
    # plt.imshow(p)
    # plt.show()
