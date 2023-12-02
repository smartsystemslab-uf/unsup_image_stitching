import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import Regularizer


class PNSRMetric(tf.keras.metrics.Metric):
    """Peak Signal-to-Noise Ratio"""

    def __init__(self, max_val=1.0, name="PNSRMetric", **kwargs):
        super(PNSRMetric, self).__init__(name=name, **kwargs)
        self.psnr = self.add_weight(name="psnr", initializer="zeros", dtype=tf.float32)
        self.count = self.add_weight(name="count", initializer="zeros", dtype=tf.float32)
        self.max_val = max_val

    def update_state(self, y_true, y_pred, sample_weight=None):
        psnr_value = tf.image.psnr(y_true, y_pred, max_val=self.max_val)
        self.psnr.assign_add(tf.reduce_sum(psnr_value))
        # self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
        self.count.assign_add(tf.cast(tf.size(psnr_value), tf.float32))

    def result(self):
        return self.psnr / self.count

    # def reset_states(self):
    #     self.psnr.assign(0)

    def get_config(self):
        basic_config = super(PNSRMetric, self).get_config()
        return {**basic_config, "max_val": self.max_val}


class SSIMMetric(tf.keras.metrics.Metric):
    """SSIM - Structural Similarity Index Measure"""

    def __init__(self, max_val=1.0, name="SSIMMetric", **kwargs):
        super(SSIMMetric, self).__init__(name=name, **kwargs)
        self.ssim = self.add_weight(name="ssim", initializer="zeros", dtype=tf.float32)
        self.count = self.add_weight(name="count", initializer="zeros", dtype=tf.float32)
        self.max_val = max_val

    def update_state(self, y_true, y_pred, sample_weight=None):
        ssim_value = tf.image.ssim(y_true, y_pred, max_val=self.max_val)
        self.ssim.assign_add(tf.reduce_sum(ssim_value))
        # self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
        self.count.assign_add(tf.cast(tf.size(ssim_value), tf.float32))

    def result(self):
        return self.ssim / self.count

    def get_config(self):
        basic_config = super(SSIMMetric, self).get_config()
        return {**basic_config, "max_val": self.max_val}


class TVRegularizer(Regularizer):
    """Enforces smoothness in image output."""

    def __init__(self, img_width, img_height, weight=1.0):
        self.img_width = img_width
        self.img_height = img_height
        self.weight = weight
        self.uses_learning_phase = False
        super(TVRegularizer, self).__init__()

    def __call__(self, x):
        assert K.ndim(x.output) == 4
        x_out = x.output

        a = K.square(
            x_out[:, : self.img_width - 1, : self.img_height - 1, :]
            - x_out[:, 1:, : self.img_height - 1, :]
        )
        b = K.square(
            x_out[:, : self.img_width - 1, : self.img_height - 1, :]
            - x_out[:, : self.img_width - 1, 1:, :]
        )
        loss = self.weight * K.mean(K.sum(K.pow(a + b, 1.25)))
        return loss


from tensorflow.keras.applications.inception_v3 import InceptionV3


def tf_cov(x):
    mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
    mx = tf.matmul(mean_x, mean_x, transpose_a=True)
    vx = tf.matmul(x, x, transpose_a=True) / tf.cast(tf.shape(x)[0], tf.float32)
    # mx = tf.matmul(tf.transpose(mean_x), mean_x)
    # vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float32)
    cov_xx = vx - mx
    return cov_xx


class FIDMetric(tf.keras.metrics.Metric):
    """FID (Fréchet inception distance) index between img1 and img2"""

    def __init__(self, name, img_shape, **kwargs):
        super(FIDMetric, self).__init__(name=name, **kwargs)
        self.fid = self.add_weight(name="fid", initializer="zeros", dtype=tf.float32)
        self.count = self.add_weight(name="count", initializer="zeros", dtype=tf.float32)
        self.img_shape = img_shape
        self.model = InceptionV3(include_top=False, pooling="avg", input_shape=img_shape)

    def update_state(self, y_true, y_pred, eps=1e-6):
        # calculate activations
        act1 = self.model(y_true)
        act2 = self.model(y_pred)
        # calculate mean and covariance statistics
        mu1, sigma1 = tf.reduce_mean(act1, axis=0, keepdims=True), tf_cov(act1)
        mu2, sigma2 = tf.reduce_mean(act2, axis=0, keepdims=True), tf_cov(act2)
        # calculate sum squared difference between means
        ssdiff = tf.reduce_sum((mu1 - mu2) ** 2.0)
        # calculate sqrt of product between cov
        covmean = tf.linalg.sqrtm(tf.matmul(sigma1, sigma2))
        # print("=====================>> act2.shape", act2.shape, mu2.shape, sigma2.shape, covmean.shape)
        #        =====================>> act2.shape (16, 2048) (1, 2048) (2048, 2048) (2048, 2048)

        if not tf.reduce_all(tf.math.is_finite(covmean)):
            # "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
            offset = tf.eye(sigma1.shape[0]) * eps
            covmean = tf.linalg.sqrtm(tf.matmul(sigma1 + offset, sigma2 + offset))

        # check and correct imaginary numbers from sqrt
        covmean = (
            tf.math.real(covmean)
            if not tf.reduce_all(tf.math.is_finite(covmean))
            else 0.0
        )
        # calculate score
        # fid_value = ssdiff + tf.reduce_sum(tf.linalg.trace(sigma1 + sigma2 - 2.0 * covmean))
        fid_value = ssdiff + tf.reduce_sum(
            tf.linalg.tensor_diag_part(sigma1 + sigma2 - 2.0 * covmean)
        )

        self.fid.assign_add(tf.reduce_sum(fid_value))
        self.count.assign_add(tf.cast(tf.size(fid_value), tf.float32))
        # self.count.assign_add(tf.cast(tf.constant(1.0), tf.float32))

    def result(self):
        return self.fid / self.count

    def get_config(self):
        basic_config = super(FIDMetric, self).get_config()
        return {**basic_config, "img_shape": self.img_shape}


class ISMetric(tf.keras.metrics.Metric):
    """Inception Score (IS) between img1 and img2"""

    def __init__(self, name, img_shape, **kwargs):
        super(ISMetric, self).__init__(name=name, **kwargs)
        self.fid = self.add_weight(name="fid", initializer="zeros", dtype=tf.float32)
        self.count = self.add_weight(name="count", initializer="zeros", dtype=tf.float32)
        self.img_shape = img_shape
        self.model = InceptionV3(include_top=False, pooling="avg", input_shape=img_shape)

    def update_state(self, y_true, eps=1e-16):
        # calculate activations
        p_yx = self.model(y_true)
        # calculate p(y)
        p_y = tf.expand_dims(tf.reduce_mean(p_yx, axis=0), 0)
        # print("=====================>> p_yx.shape", p_yx.shape, p_y.shape)
        # kl divergence for each image
        kl_d = p_yx * (tf.math.log(p_yx + eps) - tf.math.log(p_y + eps))
        # # sum over classes
        # sum_kl_d = tf.reduce_sum(kl_d, axis=1)
        # # average over images
        # avg_kl_d = tf.reduce_mean(sum_kl_d)
        # # undo the logs
        # is_score = avg_kl_d # exp(avg_kl_d)

        # average over images
        is_score = tf.reduce_mean(kl_d, axis=1)

        self.fid.assign_add(tf.reduce_sum(is_score))
        self.count.assign_add(tf.cast(1, tf.float32))

    def result(self):
        return self.fid / self.count

    def get_config(self):
        basic_config = super(ISMetric, self).get_config()
        return {**basic_config, "img_shape": self.img_shape}


class LPIPSMetric(tf.keras.metrics.Metric):
    """Learned Perceptual Image Patch Similarity (LPIPS) between img1 and img2"""

    def __init__(
        self, name, pb_fname="/home/UFAD/enghonda/.lpips/net_vgg_v0.1.pb", **kwargs
    ):
        super(LPIPSMetric, self).__init__(name=name, **kwargs)
        self.lpips = self.add_weight(name="lpips", initializer="zeros", dtype=tf.float32)
        self.count = self.add_weight(name="count", initializer="zeros", dtype=tf.float32)
        loaded = tf.saved_model.load(pb_fname)
        self.frozen_func = loaded.signatures["serving_default"]

    def update_state(self, y_true, y_pred):
        # calculate activations
        input0 = tf.transpose(y_true, [0, 3, 1, 2]) * 2.0 - 1.0
        input1 = tf.transpose(y_pred, [0, 3, 1, 2]) * 2.0 - 1.0
        distance = self.frozen_func(
            in0=tf.convert_to_tensor(input0, dtype=tf.float32),
            in1=tf.convert_to_tensor(input1, dtype=tf.float32),
        )

        self.lpips.assign_add(tf.reduce_sum(list(distance.values())[0]))
        self.count.assign_add(tf.cast(1, tf.float32))

    def result(self):
        return self.lpips / self.count

    def get_config(self):
        basic_config = super(LPIPSMetric, self).get_config()
        return {**basic_config, "img_shape": self.img_shape}
