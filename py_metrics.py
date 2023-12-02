import math

import cv2
import numpy as np
from scipy.linalg import sqrtm

CST_SEAM = "seam"  # Constant for seam-based
CST_MULB = "multiband"  # Constant for multiband blending
CST_DEEP = "ours"  # Constant for deep learning approach


def calculate_mse(img1, img2):
    # mse - img1 and img2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    return mse


def calculate_psnr(img1, img2, max_val=255.0):
    # img1 and img2 have range [0, max_val]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(max_val / math.sqrt(mse))


def ssim(img1, img2, max_val=255.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    # img1 and img2 have range [0, max_val]
    C1 = (k1 * max_val) ** 2
    C2 = (k2 * max_val) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(filter_size, filter_sigma)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def calculate_ssim(img1, img2, max_val=255.0):
    """calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, max_val]
    """
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2, max_val=max_val)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2, max_val=max_val))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2), max_val=max_val)
    else:
        raise ValueError("Wrong input image dimensions.")


# # https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf
# def calculate_msssim(img1, img2, max_val=255.0):
#     """calculate MS-SSIM
#     the same outputs as MATLAB's
#     img1, img2: [0, 255]
#     """
#     if not img1.shape == img2.shape:
#         raise ValueError("Input images must have the same dimensions.")
#     if img1.ndim == 2:
#         return msssim(img1, img2, max_val=max_val)
#     elif img1.ndim == 3:
#         if img1.shape[2] == 3:
#             l_msssim = []
#             for i in range(3):
#                 l_msssim.append(msssim(img1, img2, max_val=max_val))
#             return np.array(l_msssim).mean()
#         elif img1.shape[2] == 1:
#             return msssim(np.squeeze(img1), np.squeeze(img2), max_val=max_val)
#     else:
#         raise ValueError("Wrong input image dimensions.")


# def msssim(
#     img1,
#     img2,
#     max_val=255.0,
#     filter_size=11,
#     filter_sigma=1.5,
#     k1=0.01,
#     k2=0.03,
#     weights=None,
# ):
#     C1 = (k1 * max_val) ** 2
#     C2 = (k2 * max_val) ** 2

#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     kernel = cv2.getGaussianKernel(filter_size, filter_sigma)
#     window = np.outer(kernel, kernel.transpose())

#     mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1**2
#     mu2_sq = mu2**2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
#         (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
#     )
#     cs_map = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

#     # ssim_map = ssim_map**0.5
#     # cs_map = cs_map**0.5

#     ssim_val = ssim_map.mean()
#     cs_val = cs_map.mean()

#     return ssim_val * cs_val


class BasedMetric:
    def __init__(self) -> None:
        self.value = 0.0
        self.count = 0

    def reset(self):
        self.value = 0.0
        self.count = 0

    def update(self, *args, **kwargs):
        self.add_image(*args, **kwargs)

    def add_image(self, *args, **kwargs):
        raise NotImplementedError

    def get_value(self):
        return self.value / self.count


class UnaryMetric(BasedMetric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def add_image(self, img):
        NotImplementedError


class BinaryMetric(BasedMetric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def add_image(self, img1, img2):
        NotImplementedError


class PSNR(BinaryMetric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_val = kwargs.get("max_val", 1.0)

    def add_image(self, img1, img2):
        self.value = calculate_psnr(img1, img2, max_val=self.max_val)
        self.count += 1


class MSE(BinaryMetric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def add_image(self, img1, img2):
        self.value = calculate_mse(img1, img2)
        self.count += 1


class SSIM(BinaryMetric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_val = kwargs.get("max_val", 1.0)

    def add_image(self, img1, img2):
        self.value = calculate_ssim(img1, img2, max_val=self.max_val)
        self.count += 1


# class MSSSIM(BinaryMetric):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.max_val = kwargs.get("max_val", 1.0)

#     def add_image(self, img1, img2):
#         self.value = calculate_msssim(img1, img2, max_val=self.max_val)
#         self.count += 1


class FID(BinaryMetric):
    def __init__(self, model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.__act1 = []  # Temporary activations for image 1
        self.__act2 = []  # Temporary activations for image 2

    # calculate frechet inception distance
    def __add_image(self, images1, images2):

        if images1.ndim == 3:
            images1 = np.expand_dims(images1, 0)
        if images2.ndim == 3:
            images2 = np.expand_dims(images2, 0)

        # Assume image values are between 0 and 255.
        processed1 = images1 / 127.5 - 1.0
        processed2 = images2 / 127.5 - 1.0
        # calculate activations
        act1 = self.model.predict(processed1)
        act2 = self.model.predict(processed2)
        self.__act1.append(act1.ravel())
        self.__act2.append(act2.ravel())

    def add_image(self, image1, image2):

        if image1.shape[-1] > 3:
            nb_layers = int(image1.shape[-1] // 3)

            # Save SandFall layers
            for i in range(nb_layers):
                img = image1[:, :, :, i * 3 : (i + 1) * 3]
                mask = img != 0.0
                masked_img = mask * image2
                self.__add_image(img, masked_img)
        else:
            self.__add_image(image1, image2)

    def get_value(self):
        self.value = self.__calculate_fid()
        return self.value

    # calculate frechet inception distance
    def __calculate_fid(self):
        if not (self.__act1 and self.__act1):
            return 0
        act1 = np.array(self.__act1, dtype=np.float32)
        act2 = np.array(self.__act2, dtype=np.float32)

        # calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        # print(sigma1, sigma2.ndim)
        # calculate sqrt of product between cov
        sig = sigma1.dot(sigma2)
        if sig.ndim > 0:
            covmean = sqrtm(sig)
        else:
            covmean = np.sqrt(sig)
        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        if sig.ndim > 0:
            fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        else:
            fid = ssdiff + sigma1 + sigma2 - 2.0 * covmean
        return fid

    # set model
    def set_model(self, model):
        self.model = model

    # Reset the metric
    def reset(self):
        super().reset()
        self.__act1 = []
        self.__act2 = []


# Compute the Inception Score
class ICScore(UnaryMetric):
    def __init__(self, model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.__p_yx = []  # Temporary activations for image

    # calculate the inception score for p(y|x)
    def add_image(self, image):

        if image.ndim == 3:
            image = np.expand_dims(image, 0)

        # assume image values are between  0-255
        processed = image.astype(np.float32) / 127.5 - 1.0
        # calculate p(y|x)
        p_yx = self.model.predict(processed)

        self.__p_yx.append(p_yx.ravel())

    def get_value(self):
        self.value = self.__calculate_inception_score()
        return self.value

    # calculate the inception score for p(y|x)
    def __calculate_inception_score(self, eps=1e-16):
        if not (self.__p_yx):
            return 0
        # calculate p(y|x)
        p_yx = np.array(self.__p_yx, dtype=np.float32)
        # calculate p(y)
        p_y = np.expand_dims(p_yx.mean(axis=0), 0)
        # kl divergence for each image
        kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over images
        avg_kl_d = np.mean(sum_kl_d)
        # undo the logs
        is_score = np.exp(avg_kl_d)
        return is_score

    # set model
    def set_model(self, model):
        self.model = model

    def reset(self):
        super().reset()
        self.__p_yx = []


# Compute the Improved Inception Score(SG)
class IGScore(UnaryMetric):
    def __init__(self, model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.__p_yx = []  # Temporary activations for image

    # calculate the inception score for p(y|x)
    def add_image(self, image):

        if image.ndim == 3:
            image = np.expand_dims(image, 0)

        # assume image values are between  0-255
        processed = image.astype(np.float32) / 127.5 - 1.0
        # calculate p(y|x)
        p_yx = self.model.predict(processed)

        self.__p_yx.append(p_yx.ravel())

    def get_value(self):
        self.value = self.__calculate_s_inception_score()
        return self.value

    # calculate the inception score for p(y|x)
    def __calculate_s_inception_score(self, eps=1e-16):
        if not (self.__p_yx):
            return 0
        # calculate p(y|x)
        p_yx = np.array(self.__p_yx, dtype=np.float32)
        # calculate p(y)
        p_y = np.expand_dims(p_yx.mean(axis=0), 0)
        # kl divergence for each image
        kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over images
        avg_kl_d = np.mean(sum_kl_d)
        # undo the logs
        sg_score = avg_kl_d  # np.exp(avg_kl_d)
        return sg_score

    # set model
    def set_model(self, model):
        self.model = model

    # reset the metric
    def reset(self):
        super().reset()
        self.__p_yx = []


# Compute the Learned Perceptual Similarity Metric (LPIPS)
class LPIPS(BinaryMetric):
    """Learned Perceptual Image Patch Similarity (LPIPS) between img1 and img2"""

    def __init__(self, model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.frozen_func = model
        self.__distance = []  # Temporary activations for image

    def add_image(self, image1, image2):

        if image1.shape[-1] > 3:
            nb_layers = int(image1.shape[-1] // 3)

            # Save SandFall layers
            for i in range(nb_layers):
                img = image1[:, :, :, i * 3 : (i + 1) * 3]
                mask = img != 0.0
                masked_img = mask * image2
                self.__add_image(img, masked_img)
        else:
            self.__add_image(image1, image2)

    def __add_image(self, y_true, y_pred):

        if y_true.ndim == 3:
            y_true = np.expand_dims(y_true, 0)
        if y_pred.ndim == 3:
            y_pred = np.expand_dims(y_pred, 0)

        # calculate activations
        input0 = np.transpose(y_true, [0, 3, 1, 2]) / 127.5 - 1.0
        input1 = np.transpose(y_pred, [0, 3, 1, 2]) / 127.5 - 1.0
        distance = self.frozen_func(
            in0=tf.convert_to_tensor(input0, dtype=tf.float32),
            in1=tf.convert_to_tensor(input1, dtype=tf.float32),
        )

        self.__distance.append(distance["185"].numpy())

    # Calculate the LPIPS score
    def __calculate_lpips(self):
        return (
            np.array(self.__distance, dtype=np.float32).mean() if self.__distance else 0.0
        )

    def get_value(self):
        self.value = self.__calculate_lpips()
        return self.value

    # Reset the metric
    def reset(self):
        super().reset()
        self.__distance = []


class Metrics:
    """Metrics class"""

    def __init__(self, model, frozen_lpips_func, *args, **kwargs) -> None:  # noqa: D107
        super().__init__(*args, **kwargs)
        self.model = model
        self.frozen_lpips_func = frozen_lpips_func
        self.fid_metric = FID(model)
        self.is_metric = ICScore(model)
        self.ig_metric = IGScore(model)
        self.lpips_metric = LPIPS(frozen_lpips_func)
        # self.ssim_metric = SSIM()
        # # self.ms_ssim_metric = MSSSIM()
        # self.psnr_metric = PSNR()
        # self.mse_metric = MSE()

    def update(self, y_true, y_pred):
        self.fid_metric.update(y_true, y_pred)
        self.is_metric.update(y_pred)
        self.ig_metric.update(y_pred)
        self.lpips_metric.update(y_true, y_pred)
        # self.ssim_metric.update(y_true, y_pred)
        # # self.ms_ssim_metric.update(y_true, y_pred)
        # self.psnr_metric.update(y_true, y_pred)
        # self.mse_metric.update(y_true, y_pred)

    def get_value(self):
        return {
            "fid": self.fid_metric.get_value(),
            "is": self.is_metric.get_value(),
            "ig": self.ig_metric.get_value(),
            "lpips": self.lpips_metric.get_value(),
            # "ssim": self.ssim_metric.get_value(),
            # # "msssim": self.ms_ssim_metric.get_value(),
            # "psnr": self.psnr_metric.get_value(),
            # "mse": self.mse_metric.get_value(),
        }

    def reset(self):
        self.fid_metric.reset()
        self.is_metric.reset()
        self.ig_metric.reset()
        self.lpips_metric.reset()
        # self.ssim_metric.reset()
        # # self.ms_ssim_metric.reset()
        # self.psnr_metric.reset()
        # self.mse_metric.reset()

    # set inception model
    def set_model(self, model):
        self.model = model
        self.fid_metric.set_model(model)
        self.is_metric.set_model(model)
        self.ig_metric.set_model(model)

    # set lpips model
    def set_lpips_model(self, frozen_lpips_func):
        self.frozen_lpips_func = frozen_lpips_func
        self.lpips_metric.set_model(frozen_lpips_func)


import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3


def create_metrics(input_shape, pb_fname="/home/UFAD/enghonda/.lpips/net_vgg_v0.1.pb"):
    """Create metrics"""
    # Load InceptionV3 model
    inception_model = InceptionV3(
        include_top=False, pooling="avg", input_shape=input_shape
    )
    # Freeze the model
    inception_model.trainable = False

    # Load LPIPS model
    loaded = tf.saved_model.load(pb_fname)
    frozen_lpips_func = loaded.signatures["serving_default"]

    # Create metrics
    metrics = Metrics(inception_model, frozen_lpips_func)
    return metrics


def create_metrics_from_icmodel(
    model, pb_fname="/home/UFAD/enghonda/.lpips/net_vgg_v0.1.pb"
):
    """Create metrics from inception model"""
    inception_model = model

    # # Freeze the model
    # inception_model.trainable = False

    # Load LPIPS model
    loaded = tf.saved_model.load(pb_fname)
    frozen_lpips_func = loaded.signatures["serving_default"]

    # Create metrics
    metrics = Metrics(inception_model, frozen_lpips_func)
    return metrics


def create_metrics_from_model(inception_model, frozen_lpips_func) -> Metrics:
    """Create metrics from model"""

    # Create metrics
    metrics = Metrics(inception_model, frozen_lpips_func)
    return metrics


def main():
    """Main function"""
    input_shape = (299, 299, 3)

    # Create metrics
    metrics = create_metrics(input_shape)

    # Load images
    img1 = np.random.rand(*input_shape)
    img2 = np.random.rand(*input_shape)

    # Add images
    metrics.update(img1, img2)

    # Get metrics
    print(metrics.get_value())


if __name__ == "__main__":
    main()
