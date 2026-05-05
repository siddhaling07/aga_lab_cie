import argparse
import os

from .data import load_creditcard_data, prepare_training_data
from .model import build_generator, build_discriminator, build_gan, build_anogan_discriminator
from .trainer import train_gan, train_anogan
from .evaluator import (
    compute_anomaly_score_gan,
    compute_anomaly_score_anogan,
    determine_threshold,
    predict_anomalies,
    evaluate_predictions,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a GAN-based fraud detection model."
    )
    parser.add_argument(
        "csv_path",
        help="Path to the creditcard.csv dataset",
    )
    parser.add_argument(
        "--model",
        choices=["gan", "anogan"],
        default="anogan",
        help="GAN model type: 'gan' for full GAN, 'anogan' for AnoGAN",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=100,
        help="Latent dimension for GAN",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=99.5,
        help="Percentile for anomaly threshold using normal transaction errors",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"Dataset not found at {args.csv_path}")

    X_scaled, y, _ = load_creditcard_data(args.csv_path)
    X_train, X_test, y_test = prepare_training_data(X_scaled, y)

    if args.model == "gan":
        generator = build_generator(input_dim=X_train.shape[1], latent_dim=args.latent_dim)
        discriminator = build_discriminator(input_dim=X_train.shape[1])
        discriminator.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        gan = build_gan(generator, discriminator)
        gan.compile(optimizer="adam", loss="binary_crossentropy")
        train_gan(generator, discriminator, gan, X_train, epochs=args.epochs, batch_size=args.batch_size)
        anomaly_scores = compute_anomaly_score_gan(generator, discriminator, X_test, latent_dim=args.latent_dim)
    elif args.model == "anogan":
        discriminator = build_anogan_discriminator(input_dim=X_train.shape[1])
        train_anogan(generator=None, discriminator=discriminator, X_train=X_train, epochs=args.epochs, batch_size=args.batch_size)
        anomaly_scores = compute_anomaly_score_anogan(discriminator, X_test)

    threshold = determine_threshold(anomaly_scores, y_test, percentile=args.threshold_percentile)
    y_pred = predict_anomalies(anomaly_scores, threshold)
    metrics = evaluate_predictions(y_test, y_pred)

    print("Threshold: {:.6f}".format(threshold))
    print("Precision: {:.4f}".format(metrics["precision"]))
    print("Recall: {:.4f}".format(metrics["recall"]))
    print("F1 Score: {:.4f}".format(metrics["f1_score"]))
    print(metrics["classification_report"])


if __name__ == "__main__":
    main()
