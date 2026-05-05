"""Fraud GAN package."""

from .data import load_creditcard_data, prepare_training_data
from .model import build_generator, build_discriminator, build_gan, build_anogan_discriminator
from .trainer import train_gan, train_anogan, train_autoencoder
from .evaluator import (
    compute_anomaly_score_gan,
    compute_anomaly_score_anogan,
    compute_reconstruction_error,
    determine_threshold,
    predict_anomalies,
    evaluate_predictions,
)

__all__ = [
    "load_creditcard_data",
    "prepare_training_data",
    "build_generator",
    "build_discriminator",
    "build_gan",
    "build_anogan_discriminator",
    "train_gan",
    "train_anogan",
    "train_autoencoder",
    "compute_anomaly_score_gan",
    "compute_anomaly_score_anogan",
    "compute_reconstruction_error",
    "determine_threshold",
    "predict_anomalies",
    "evaluate_predictions",
]
