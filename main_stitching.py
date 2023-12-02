import argparse
import os
import sys
import time

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3

import constant as cfg
import model_stitching
import panowrapper as pw
import SSL_models
from py_metrics import Metrics, create_metrics_from_model

pb_fname = "/home/UFAD/enghonda/.lpips/net_vgg_v0.1.pb"
# Load LPIPS model
loaded = tf.saved_model.load(pb_fname)
frozen_lpips_func = loaded.signatures["serving_default"]
metrics_seam: Metrics = create_metrics_from_model(
    inception_model=None, frozen_lpips_func=frozen_lpips_func
)
metrics_deep: Metrics = create_metrics_from_model(
    inception_model=None, frozen_lpips_func=frozen_lpips_func
)
metrics_mbb: Metrics = create_metrics_from_model(
    inception_model=None, frozen_lpips_func=frozen_lpips_func
)


def save_blending_result(img_merge, outdir, file_prefix, nb_layers=None):
    img_shape = img_merge.shape
    if len(img_shape) < 4:
        cfg.PRINT_ERROR("Invalid Image Shape")
        exit(1)

    if not nb_layers:
        nb_layers = int(img_shape[3] // 3)

    # Save SandFall layers
    for i in range(nb_layers):
        filename = os.path.join(outdir, file_prefix + str(i) + ".jpg")
        img = img_merge[0, :, :, i * 3 : (i + 1) * 3]
        cv2.imwrite(
            filename, cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        )


def stitch(files, model_type, outdir, scale_factor, compare_result=True):
    global panow
    global model
    os.makedirs(m_outdir, exist_ok=True)
    suffix = time.strftime("_%Y%m%d-%H%M%S")
    img_merge = panow.pano_stitch_single_camera(
        files, multi_band_blend=-1, return_img=True
    )
    save_blending_result(
        img_merge,
        outdir,
        "merge_layer",
        min(abs(cfg.sandfall_layer), img_merge.shape[-1] // 3),
    )
    sandfall_block = panow.pano_stitch_single_camera(
        files, multi_band_blend=-5, return_img=True
    )
    # panow.print_config()
    if sandfall_block is None:
        print(f"failed to stitch the images {files}")
        sys.exit()

    # Save SandFall layers
    save_blending_result(
        sandfall_block, outdir, "sandfall_layer", abs(cfg.sandfall_layer)
    )

    img_mbb = None
    if compare_result:
        out_filename = os.path.join(outdir, "multiband_20_" + suffix + ".jpg")
        img_mbb = panow.pano_stitch_single_camera(
            files, out_filename=out_filename, multi_band_blend=20, return_img=True
        )
        img_mbb = img_mbb.astype(np.float32) * 255.0
        img_mbb = np.clip(img_mbb, 0, 255).astype("uint8")
        img_mbb = img_mbb[0, :, :, :]

        out_seam_img = os.path.join(outdir, "seam_blending_" + suffix + ".jpg")
        img_seam = panow.pano_stitch_single_camera(
            files, out_filename=out_seam_img, multi_band_blend=-25, return_img=True
        )
        img_seam = img_seam.astype(np.float32) * 255.0
        img_seam = np.clip(img_seam, 0, 255).astype("uint8")
        img_seam = img_seam[0, :, :, :]

        print(
            f"=> Shape im_merge: {sandfall_block.shape}, shape of img_mbb: {img_mbb.shape}, shape of img_seam: {img_seam.shape}"
        )

    # Final image size
    final_h, final_w = sandfall_block.shape[1], sandfall_block.shape[2]
    # if model_type == "ddis" or model_type == "unddis" or model_type == "unetis":
    if model_type in ["ddis", "unddis", "unetis"]:
        # Pad image with zeros to the nearest power of 2
        h = nearest_mult_n(sandfall_block.shape[1], 16) - sandfall_block.shape[1]
        w = nearest_mult_n(sandfall_block.shape[2], 16) - sandfall_block.shape[2]
        sandfall_block = np.pad(
            sandfall_block, ((0, 0), (0, h), (0, w), (0, 0)), mode="constant"
        )

    start_time = time.time()
    result = model.simple_stitch(
        sandfall_block,
        out_dir=outdir,
        scale_factor=scale_factor,
        suffix=model_type + str(suffix),
        return_image=True,
    )
    # Crop the image to the original size
    result = result[:final_h, :final_w, :]
    print("--- %s seconds ---" % (time.time() - start_time))

    if compare_result:
        cfg.PRINT_INFO(
            f"Computing Metrics - im_mg.shape: {img_merge.shape}, result.shape: {result.shape}"
        )
        inceptionv3_model = InceptionV3(
            include_top=False, pooling="avg", input_shape=result.shape
        )
        # Freeze the model
        inceptionv3_model.trainable = False

        # set the inception model to the metrics
        metrics_seam.set_model(inceptionv3_model)
        metrics_mbb.set_model(inceptionv3_model)
        metrics_deep.set_model(inceptionv3_model)

        # update the metrics with new images
        metrics_deep.update(img_merge, result)
        metrics_mbb.update(img_merge, img_mbb)
        metrics_seam.update(img_merge, img_seam)


parser = argparse.ArgumentParser(
    description="Up-Scales an image using Image Super Resolution Model"
)
# parser.add_argument("imgpath", type=str, nargs="*", help="Path to input image")
parser.add_argument(
    "--imgdir", nargs="?", type=str, default=None, help="Image directory"
)
parser.add_argument("--outdir", type=str, default=None, help="Output Result directory")
parser.add_argument(
    "--calib_dir",
    nargs="?",
    type=str,
    default=None,
    help="Camera Calibration directory that contains images",
)
parser.add_argument(
    "--calib_pattern",
    nargs="?",
    type=str,
    default=None,
    help="Calibration images pattern. Should contains the camera id and image id."
    "Ex: 'DIR/Input/{camID:05d}/{imgID:05d}.jpg'",
)
parser.add_argument(
    "--input_pattern",
    nargs="?",
    type=str,
    default=None,
    help="Input image pattern contain the string to retrieved images. Should contains the camera id and image id."
    "Ex: 'DIR/Input/{camID:05d}/{imgID:05d}.jpg'",
)
parser.add_argument(
    "-nbc",
    "--nb_cameras",
    nargs="?",
    type=int,
    default=0,
    help="Total number of cameras",
)
parser.add_argument(
    "-nbi",
    "--nb_images",
    nargs="?",
    type=int,
    default=0,
    help="Maximun number of images to use for calibration",
)
parser.add_argument(
    "-nbis",
    "--nb_stitch_images",
    nargs="?",
    type=int,
    default=0,
    help="Maximun number of images to stitch in the pattern",
)
parser.add_argument(
    "--compare_result",
    nargs="?",
    default=True,
    type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
)
parser.add_argument(
    "--scale_factor",
    nargs="?",
    type=float,
    default=1.0,
    help="Input image scale factor [to be divide]."
    "For example use 3 to scale the input images by a factor of 1/3",
)

parser.add_argument(
    "--files",
    nargs="+",
    help="List of image file to stitch",
    metavar="file1.jpg file2.jpg file3.png",
)
parser.add_argument(
    "--dfs",
    nargs="?",
    type=str,
    default="MCMI",
    choices=["MCMI", "SCMI", "MCSI", "LIST", "IDIR"],
    help="Dataset File Structure (DFS) \n"
    "MCMI: Multi-Camera Multi-Image (indicate both the 'camID' and the 'imgID' in the file pattern \n"
    "SCMI: Single-Camera Multi-Image ('imgID' in the file pattern"
    "MCSI: Multi-Camera Multi-Image ('camID' in the file pattern"
    "LIST: list of files image to stitch and provide the list in the --files param"
    "IDIR: Image Directory of files with *.jpg, *.jpeg, *.png, and *.bmp will be retrieve in the folder indicate by --imgdir",
)
parser.add_argument(
    "--metric",
    type=str,
    default=None,
    help="Pretrained Metric Weight to use: "
    "lpips: Learned Perceptual Image Patch Similarity"
    "is: Inception Score (IS)"
    "pnsr: Peak Signal-to-Noise Ratio"
    "ssim: SSIM - Structural Similarity Index Measure ",
)

parser.add_argument(
    "--model",
    type=str,
    default="ddis",
    help="Use either image super resolution (is), "
    "expanded super resolution (eis), "
    "denoising auto encoder img stitching (dis), "
    "deep denoising img stitching (ddis) or res net sr (rnis)",
)


def nearest_power_2(x: int):
    return 1 << (x - 1).bit_length()


def nearest_mult_n(m: int, n: int = 4):
    return m if m % n == 0 else ((m // n) + 1) * n


args = parser.parse_args()
print(args)
model_type = str(args.model).lower()
if not model_type in [
    "is",
    "eis",
    "dis",
    "ddis",
    "rnis",
    "distilled_rnis",
    "unrnis",
    "unddis",
    "unetis",
]:
    raise ValueError(
        'Model type must be either "is", "eis", "dis", '
        '"ddis", "rnis" or "distilled_rnis", "unrnis", "unddis", "unetis"'
    )

panow = None
model = None

if __name__ == "__main__":
    start_time = time.time()

    if args.imgdir is None or args.input_pattern is None:
        ValueError(
            "Please provide the list of files or the directory or the pattern containing the images to be stitched"
        )

    files = []
    if args.dfs == "MCMI":
        files = [
            [
                args.input_pattern.format(camID=i, imgID=img_id)
                for i in range(args.nb_cameras)
            ]
            for img_id in range(args.nb_stitch_images)
        ]
    elif args.dfs == "SCMI":
        files = [
            args.input_pattern.format(imgID=img_id)
            for img_id in range(args.nb_stitch_images)
        ]
    elif args.dfs == "MCSI":
        files = [args.input_pattern.format(camID=i) for i in range(args.nb_cameras)]
    elif args.dfs == "LIST":
        files = args.files
    elif args.dfs == "IDIR":
        import re

        # exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]
        exts = ["jpg", "jpeg", "png", "bmp"]
        for ext in exts:
            files.extend(
                [
                    os.path.join(args.imgdir, filename)
                    for filename in os.listdir(args.imgdir)
                    if re.search(r"\." + ext + "$", filename, re.IGNORECASE)
                ]
            )
            # files.extend(glob.glob(os.path.join(args.imgdir, ext)))
        print(files)
    else:
        ValueError("Please indicate the dataset file structure for your images")

    if model_type == "is":  # Work
        model = model_stitching.ImageStitchingModel()
    elif model_type == "eis":  # Work
        model = model_stitching.ExpantionStitching()
    elif model_type == "dis":  # Do Not Work
        model = model_stitching.DenoisingAutoEncoderStitch()
    elif model_type == "ddis":  # Do Not Work
        model = model_stitching.DeepDenoiseStitch()
    elif model_type == "rnis":  # Work
        model = model_stitching.ResNetStitch()
    elif model_type == "unrnis":  # Work
        model = SSL_models.ResNetStitch(metric=args.metric)
    elif model_type == "unddis":  # Work
        model = SSL_models.DeepDenoiseStitch(metric=args.metric)
    elif model_type == "unetis":  # Work
        model = SSL_models.U_NetStitch(metric=args.metric)
    elif model_type == "distilled_rnis":  # Not Trained Yet
        # model = model_stitching.DistilledResNetStitch()
        print("The Distilled model has not been trained yet.")
        sys.exit()
    else:
        model = model_stitching.ImageStitchingModel()

    panow = pw.PanoWrapper(
        scale_factor_x=args.scale_factor, scale_factor_y=args.scale_factor
    )

    ## Check for image parameters
    # initialize pano stitch
    if args.calib_pattern and args.nb_cameras and args.nb_images:
        panow.init_pano_from_dir(args.calib_pattern, args.nb_cameras, args.nb_images)

    elif not panow.is_pano_initialize() and args.calib_dir is not None:
        panow.init_pano_stitcher(args.calib_dir, multi_band_blend=0)

    elif not panow.is_pano_initialize():
        panow.init_pano_stitcher(files, multi_band_blend=0)

    if not panow.is_pano_initialize():
        cfg.PRINT_ERROR("Failed to initialized pano object")
        sys.exit()

    m_outdir = (
        f"{cfg.dataset_folder}/out_result" if args.outdir is None else args.outdir
    )

    if args.dfs == "MCMI":
        for idx, file_list in enumerate(files):
            # file_list = [args.input_pattern.format(camID=i, imgID=img_id) for i in range(args.nb_cameras)]
            print(f"==> Processing Image {idx+1}/{len(files)}")
            stitch(
                file_list,
                model_type=model_type,
                outdir=m_outdir,
                compare_result=args.compare_result,
                scale_factor=args.scale_factor,
            )

    else:
        stitch(
            files,
            model_type=model_type,
            outdir=m_outdir,
            compare_result=args.compare_result,
            scale_factor=args.scale_factor,
        )

    if args.compare_result:
        print(
            f"deepL: {metrics_deep.get_value()}, mbb: {metrics_mbb.get_value()}, seam: {metrics_seam.get_value()} "
        )

    print(f"Total time: {time.time() - start_time:.3f} (s)")
