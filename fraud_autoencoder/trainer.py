import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


def train_gan(generator: Model, discriminator: Model, gan: Model, X_train, epochs: int = 100, batch_size: int = 64, verbose: int = 1):
    """Train the GAN on normal transactions."""
    for epoch in range(epochs):
        # Train discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]
        noise = np.random.normal(0, 1, (batch_size, generator.input_shape[1]))
        fake_imgs = generator.predict(noise, verbose=0)
        d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train generator
        noise = np.random.normal(0, 1, (batch_size, generator.input_shape[1]))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch}: D loss: {d_loss[0]:.4f}, G loss: {g_loss:.4f}")


def train_anogan(generator: Model, discriminator: Model, X_train, epochs: int = 50, batch_size: int = 256, verbose: int = 1):
    """Train AnoGAN discriminator on normal data (only real samples, no generator)."""
    # For AnoGAN, discriminator is trained only on normal data
    labels = np.ones((X_train.shape[0], 1))  # All normal
    early_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True, verbose=0)
    discriminator.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    discriminator.fit(X_train, labels, epochs=epochs, batch_size=batch_size, callbacks=[early_stop], verbose=verbose)


def train_autoencoder(
    model: Model,
    X_train,
    epochs: int = 20,
    batch_size: int = 256,
    validation_split: float = 0.1,
    verbose: int = 2,
):
    """Train the autoencoder on normal transactions only."""
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )
    history = model.fit(
        X_train,
        X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=verbose,
    )
    return history
