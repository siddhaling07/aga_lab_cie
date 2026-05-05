from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam


def build_generator(input_dim: int, latent_dim: int = 100) -> Model:
    """Build the generator for GAN."""
    inputs = Input(shape=(latent_dim,))
    x = Dense(128)(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization()(x)
    outputs = Dense(input_dim, activation="tanh")(x)
    return Model(inputs, outputs, name="generator")


def build_discriminator(input_dim: int) -> Model:
    """Build the discriminator for GAN."""
    inputs = Input(shape=(input_dim,))
    x = Dense(512)(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.2)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    return Model(inputs, outputs, name="discriminator")


def build_gan(generator: Model, discriminator: Model) -> Model:
    """Build the combined GAN model."""
    discriminator.trainable = False
    inputs = Input(shape=(generator.input_shape[1],))
    generated = generator(inputs)
    validity = discriminator(generated)
    return Model(inputs, validity, name="gan")


def build_anogan_discriminator(input_dim: int) -> Model:
    """Build discriminator for AnoGAN anomaly detection."""
    inputs = Input(shape=(input_dim,))
    x = Dense(64, activation="relu")(inputs)
    x = Dense(32, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(8, activation="relu")(x)
    x = Dense(4, activation="relu")(x)
    intermediate = Dense(1, activation="sigmoid")(x)
    return Model(inputs, intermediate, name="anogan_discriminator")
