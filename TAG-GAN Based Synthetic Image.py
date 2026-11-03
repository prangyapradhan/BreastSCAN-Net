import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import Model


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = 224
CHANNELS = 1
LATENT_DIM = 128


# ============================================================
# TEXTURE ATTENTION MODULE
# ============================================================

class TextureAttention(layers.Layer):
    """
    Texture Attention module.

    Learns spatially important texture regions from MRI
    feature representations.

    Input:
        Feature map [B, H, W, C]

    Output:
        Texture-attended feature map
    """

    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)

        self.texture_conv = layers.Conv2D(
            channels,
            kernel_size=3,
            padding="same",
            activation="relu"
        )

        self.attention_conv = layers.Conv2D(
            1,
            kernel_size=1,
            padding="same",
            activation="sigmoid"
        )

    def call(self, x):

        # Local texture representation
        texture_features = self.texture_conv(x)

        # Spatial attention
        attention_map = self.attention_conv(
            texture_features
        )

        # Attention-guided texture features
        attended_features = (
            texture_features * attention_map
        )

        return attended_features


# ============================================================
# GENERATOR
# ============================================================

def build_generator(
    latent_dim=LATENT_DIM,
    channels=CHANNELS
):
    """
    Build TAG-GAN generator.

    Noise vector
        ↓
    Dense projection
        ↓
    7 x 7 feature representation
        ↓
    Upsampling
        ↓
    Texture Attention
        ↓
    Synthetic MRI
    """

    noise = layers.Input(
        shape=(latent_dim,),
        name="latent_noise"
    )

    # Project latent vector
    x = layers.Dense(
        7 * 7 * 256,
        use_bias=False
    )(noise)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Reshape(
        (7, 7, 256)
    )(x)

    # 7 -> 14
    x = layers.Conv2DTranspose(
        128,
        kernel_size=4,
        strides=2,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Texture attention
    x = TextureAttention(128)(
        x
    )

    # 14 -> 28
    x = layers.Conv2DTranspose(
        64,
        kernel_size=4,
        strides=2,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Texture attention
    x = TextureAttention(64)(
        x
    )

    # 28 -> 56
    x = layers.Conv2DTranspose(
        32,
        kernel_size=4,
        strides=2,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 56 -> 112
    x = layers.Conv2DTranspose(
        16,
        kernel_size=4,
        strides=2,
        padding="same",
        use_bias=False
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 112 -> 224
    output = layers.Conv2DTranspose(
        channels,
        kernel_size=4,
        strides=2,
        padding="same",
        activation="tanh",
        name="synthetic_mri"
    )(x)

    return Model(
        noise,
        output,
        name="TAG_GAN_Generator"
    )


# ============================================================
# DISCRIMINATOR
# ============================================================

def build_discriminator(
    img_size=IMG_SIZE,
    channels=CHANNELS
):
    """
    Build TAG-GAN discriminator.

    The discriminator receives:
        Real MRI
        OR
        Synthetic MRI

    and predicts real/fake probability.
    """

    image = layers.Input(
        shape=(img_size, img_size, channels),
        name="mri_input"
    )

    x = layers.Conv2D(
        64,
        kernel_size=4,
        strides=2,
        padding="same"
    )(image)

    x = layers.LeakyReLU(
        negative_slope=0.2
    )(x)

    x = layers.Conv2D(
        128,
        kernel_size=4,
        strides=2,
        padding="same"
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(
        negative_slope=0.2
    )(x)

    # Texture attention
    x = TextureAttention(128)(
        x
    )

    x = layers.Conv2D(
        256,
        kernel_size=4,
        strides=2,
        padding="same"
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(
        negative_slope=0.2
    )(x)

    x = layers.Conv2D(
        512,
        kernel_size=4,
        strides=2,
        padding="same"
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(
        negative_slope=0.2
    )(x)

    x = layers.GlobalAveragePooling2D()(x)

    output = layers.Dense(
        1,
        activation="sigmoid",
        name="real_fake"
    )(x)

    return Model(
        image,
        output,
        name="TAG_GAN_Discriminator"
    )


# ============================================================
# TAG-GAN MODEL
# ============================================================

class TAGGAN(tf.keras.Model):

    def __init__(
        self,
        generator,
        discriminator,
        latent_dim=LATENT_DIM
    ):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim

        self.bce = tf.keras.losses.BinaryCrossentropy()

    def compile(
        self,
        g_optimizer,
        d_optimizer
    ):

        super().compile()

        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    # --------------------------------------------------------
    # DISCRIMINATOR LOSS
    # --------------------------------------------------------

    def discriminator_loss(
        self,
        real_output,
        fake_output
    ):

        real_loss = self.bce(
            tf.ones_like(real_output),
            real_output
        )

        fake_loss = self.bce(
            tf.zeros_like(fake_output),
            fake_output
        )

        return real_loss + fake_loss

    # --------------------------------------------------------
    # GENERATOR LOSS
    # --------------------------------------------------------

    def generator_loss(
        self,
        fake_output
    ):

        return self.bce(
            tf.ones_like(fake_output),
            fake_output
        )

    # --------------------------------------------------------
    # TRAINING STEP
    # --------------------------------------------------------

    @tf.function
    def train_step(self, real_images):

        batch_size = tf.shape(
            real_images
        )[0]

        # Generate random latent vectors
        noise = tf.random.normal(
            shape=(
                batch_size,
                self.latent_dim
            )
        )

        # -----------------------------
        # Generator forward pass
        # -----------------------------

        with tf.GradientTape() as g_tape:

            fake_images = self.generator(
                noise,
                training=True
            )

            fake_output = self.discriminator(
                fake_images,
                training=True
            )

            g_loss = self.generator_loss(
                fake_output
            )

        # Generator update
        g_gradients = g_tape.gradient(
            g_loss,
            self.generator.trainable_variables
        )

        self.g_optimizer.apply_gradients(
            zip(
                g_gradients,
                self.generator.trainable_variables
            )
        )

        # -----------------------------
        # Discriminator forward pass
        # -----------------------------

        with tf.GradientTape() as d_tape:

            real_output = self.discriminator(
                real_images,
                training=True
            )

            fake_output = self.discriminator(
                fake_images,
                training=True
            )

            d_loss = self.discriminator_loss(
                real_output,
                fake_output
            )

        # Discriminator update
        d_gradients = d_tape.gradient(
            d_loss,
            self.discriminator.trainable_variables
        )

        self.d_optimizer.apply_gradients(
            zip(
                d_gradients,
                self.discriminator.trainable_variables
            )
        )

        return {
            "generator_loss": g_loss,
            "discriminator_loss": d_loss
        }


# ============================================================
# CREATE TAG-GAN
# ============================================================

def create_tag_gan():

    generator = build_generator()

    discriminator = build_discriminator()

    tag_gan = TAGGAN(
        generator,
        discriminator
    )

    tag_gan.compile(
        g_optimizer=tf.keras.optimizers.Adam(
            learning_rate=2e-4,
            beta_1=0.5
        ),
        d_optimizer=tf.keras.optimizers.Adam(
            learning_rate=2e-4,
            beta_1=0.5
        )
    )

    return tag_gan


# ============================================================
# SYNTHETIC MRI GENERATION
# ============================================================

def generate_synthetic_images(
    generator,
    number_of_images,
    output_dir,
    latent_dim=LATENT_DIM
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    noise = tf.random.normal(
        shape=(
            number_of_images,
            latent_dim
        )
    )

    synthetic_images = generator(
        noise,
        training=False
    )

    # Convert [-1, 1] -> [0, 255]
    synthetic_images = (
        (synthetic_images + 1.0)
        * 127.5
    )

    synthetic_images = tf.clip_by_value(
        synthetic_images,
        0,
        255
    )

    synthetic_images = tf.cast(
        synthetic_images,
        tf.uint8
    )

    for i, image in enumerate(
        synthetic_images
    ):

        image = image.numpy()

        if image.shape[-1] == 1:
            image = image[:, :, 0]

        import cv2

        cv2.imwrite(
            os.path.join(
                output_dir,
                f"synthetic_mri_{i+1:05d}.png"
            ),
            image
        )

    return synthetic_images


# ============================================================
# EXAMPLE TRAINING
# ============================================================

if __name__ == "__main__":

    # Create TAG-GAN
    tag_gan = create_tag_gan()

    print(
        "TAG-GAN initialized successfully."
    )

    tag_gan.generator.summary()

    tag_gan.discriminator.summary()

    # --------------------------------------------------------
    # IMPORTANT:
    # Load ONLY training-fold MRI images here.
    #
    # Example:
    #
    # train_images = np.load(
    #     "data/train_mri.npy"
    # )
    #
    # train_images = train_images.astype(
    #     np.float32
    # ) / 127.5 - 1.0
    #
    # tag_gan.fit(
    #     train_images,
    #     epochs=100,
    #     batch_size=16
    # )
    #
    # --------------------------------------------------------

    print(
        "Load patient-wise training-fold MRI data "
        "before calling model.fit()."
    )
