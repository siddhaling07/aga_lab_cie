import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from typing import Dict, Tuple
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


def compute_anomaly_score_anogan(discriminator: Model, X_test: np.ndarray) -> np.ndarray:
    """Compute anomaly score for AnoGAN: 1 - probability of being normal."""
    probs = discriminator.predict(X_test, verbose=0).flatten()
    return 1 - probs  # Higher score = more anomalous


def compute_anomaly_score_gan(generator: Model, discriminator: Model, X_test: np.ndarray, latent_dim: int = 100, steps: int = 20, lr: float = 0.1) -> np.ndarray:
    """Compute anomaly score for full GAN using latent optimization (optimized for speed)."""
    anomaly_scores = []
    
    # Process in batches to show progress
    batch_size = 100  # Process 100 samples at a time
    for i in range(0, len(X_test), batch_size):
        batch_X = X_test[i:i+batch_size]
        batch_scores = []
        
        for x in batch_X:
            # Initialize latent vector
            z = tf.Variable(tf.random.normal([1, latent_dim]), trainable=True)
            optimizer = Adam(learning_rate=lr)
            
            # Fewer optimization steps for speed
            for _ in range(steps):
                with tf.GradientTape() as tape:
                    generated = generator(z, training=False)
                    residual_loss = tf.reduce_mean(tf.square(x - generated))
                    disc_score = discriminator(generated, training=False)[0][0]
                    disc_loss = 1 - disc_score
                    loss = residual_loss + disc_loss
                grads = tape.gradient(loss, [z])
                optimizer.apply_gradients(zip(grads, [z]))
            
            # Final anomaly score
            generated = generator(z, training=False)
            residual_loss = np.mean((x - generated.numpy()[0]) ** 2)
            disc_score = discriminator(generated, training=False)[0][0].numpy()
            disc_loss = 1 - disc_score
            anomaly_score = residual_loss + disc_loss
            batch_scores.append(anomaly_score)
        
        anomaly_scores.extend(batch_scores)
        print(f"Processed {min(i+batch_size, len(X_test))}/{len(X_test)} samples...")
    
    return np.array(anomaly_scores)


def compute_reconstruction_error(model, X_test) -> np.ndarray:
    X_pred = model.predict(X_test)
    return np.mean(np.square(X_test - X_pred), axis=1)


def determine_threshold(scores: np.ndarray, y_test: np.ndarray, percentile: float = 99.5) -> float:
    return np.percentile(scores[y_test == 0], percentile)


def predict_anomalies(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores > threshold).astype(int)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred),
    }
