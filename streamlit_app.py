import os

import numpy as np
import pandas as pd
import streamlit as st

from fraud_autoencoder.data import load_creditcard_data, prepare_training_data
from fraud_autoencoder.evaluator import (
    compute_anomaly_score_gan,
    determine_threshold,
    evaluate_predictions,
    predict_anomalies,
)
from fraud_autoencoder.model import build_generator, build_discriminator, build_gan, build_anogan_discriminator
from fraud_autoencoder.trainer import train_gan, train_anogan


st.set_page_config(
    page_title="Fraud Detection GAN",
    page_icon="🛡️",
    layout="wide",
)


def explain_results(metrics: dict, threshold: float, fraud_ratio: float, fraud_count: int, predicted_fraud: int, model_type: str) -> None:
    st.markdown(
        "### Why this output matters"
    )
    if model_type == "gan":
        st.markdown(
            "- The GAN is trained on normal transactions to generate realistic data."
        )
        st.markdown(
            "- Anomalies (fraud) are detected by how well the discriminator distinguishes real from generated data and reconstruction residuals."
        )
    elif model_type == "anogan":
        st.markdown(
            "- AnoGAN trains a discriminator on normal data and detects anomalies based on how 'abnormal' the input appears to the discriminator."
        )
    st.markdown(
        "- Fraudulent transactions tend to have higher anomaly scores because they differ from normal transaction patterns."
    )
    st.markdown(
        f"- The selected threshold is **{threshold:.6f}**, meaning any transaction with anomaly score above this value is flagged as suspicious."
    )
    st.markdown(
        f"- The dataset contains **{fraud_count}** fraud cases ({fraud_ratio:.2%} of all samples), and the model predicted **{predicted_fraud}** suspicious transactions."
    )
    st.markdown(
        "- Precision measures how many predicted frauds were correct, recall measures how many true frauds were found, and F1 balances both."
    )


def display_dataset_summary(y: np.ndarray) -> None:
    st.subheader("Dataset summary")
    counts = pd.Series(y).value_counts().sort_index()
    counts.index = ["Normal", "Fraud"]
    st.write(counts)
    st.bar_chart(counts)
    st.markdown(
        "The dataset is imbalanced, so the model is trained on normal transactions only and uses anomaly detection to find fraud."
    )


def display_anomaly_distribution(scores: np.ndarray, y_test: np.ndarray, threshold: float) -> None:
    anomalies = pd.DataFrame(
        {
            "Anomaly score": scores,
            "Label": np.where(y_test == 0, "Normal", "Fraud"),
        }
    )
    st.subheader("Anomaly score distribution")
    st.write(
        "The histogram below shows how the anomaly score differs between normal and fraud transactions. "
        "A higher score usually indicates an anomalous transaction."
    )
    st.bar_chart(anomalies.groupby("Label").mean())
    st.markdown(f"**Threshold used:** {threshold:.6f}")


def main() -> None:
    st.title("Fraud Detection with GAN")
    st.markdown(
        "This app trains a GAN (Generative Adversarial Network) on normal credit card transactions and flags suspicious activity based on anomaly scores."
    )

    with st.sidebar:
        st.header("Configuration")
        csv_path = st.text_input("Dataset path", "data/creditcard.csv")
        model_type = st.selectbox("Model type", ["anogan", "gan"], index=0, 
                                help="AnoGAN: Fast, trains only discriminator. GAN: Slower but more sophisticated.")
        epochs = st.slider("Epochs", 5, 50, 10, 5)  # Reduced default epochs
        batch_size = st.selectbox("Batch size", [32, 64, 128, 256], index=1)
        latent_dim = st.slider("Latent dimension (for GAN)", 50, 200, 100, 10) if model_type == "gan" else 0
        threshold_percentile = st.slider("Threshold percentile", 90.0, 99.9, 99.5, 0.1)
        
        st.info("💡 **Tip**: AnoGAN is much faster than full GAN. Use AnoGAN for quick results!")
        
        run_button = st.button("Train and evaluate")

    if not os.path.exists(csv_path):
        st.warning("Dataset not found. Please provide a valid path to creditcard.csv.")
        return

    if run_button:
        with st.spinner("Loading dataset..."):
            X_scaled, y, _ = load_creditcard_data(csv_path)
            X_train, X_test, y_test = prepare_training_data(X_scaled, y)

        display_dataset_summary(y_test)

        with st.spinner("Training GAN..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            if model_type == "gan":
                status_text.text("Building GAN models...")
                generator = build_generator(input_dim=X_train.shape[1], latent_dim=latent_dim)
                discriminator = build_discriminator(input_dim=X_train.shape[1])
                discriminator.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                gan = build_gan(generator, discriminator)
                gan.compile(optimizer="adam", loss="binary_crossentropy")
                
                status_text.text("Training GAN...")
                train_gan(generator, discriminator, gan, X_train, epochs=epochs, batch_size=batch_size, verbose=0)
                
                status_text.text("Computing anomaly scores (this may take a few minutes)...")
                anomaly_scores = compute_anomaly_score_gan(generator, discriminator, X_test, latent_dim=latent_dim, steps=10, lr=0.1)
                
            elif model_type == "anogan":
                status_text.text("Building AnoGAN discriminator...")
                discriminator = build_anogan_discriminator(input_dim=X_train.shape[1])
                
                status_text.text("Training AnoGAN...")
                train_anogan(None, discriminator, X_train, epochs=epochs, batch_size=batch_size, verbose=0)
                
                status_text.text("Computing anomaly scores...")
                anomaly_scores = compute_anomaly_score_anogan(discriminator, X_test)
            
            progress_bar.empty()
            status_text.empty()

        with st.spinner("Evaluating model..."):
            threshold = determine_threshold(anomaly_scores, y_test, percentile=threshold_percentile)
            y_pred = predict_anomalies(anomaly_scores, threshold)
            metrics = evaluate_predictions(y_test, y_pred)

        st.subheader("Performance metrics")
        cols = st.columns(3)
        cols[0].metric("Precision", f"{metrics['precision']:.4f}")
        cols[1].metric("Recall", f"{metrics['recall']:.4f}")
        cols[2].metric("F1 Score", f"{metrics['f1_score']:.4f}")

        st.markdown("### Detailed classification report")
        st.text(metrics["classification_report"])

        display_anomaly_distribution(anomaly_scores, y_test, threshold)

        explain_results(
            metrics,
            threshold,
            fraud_ratio=np.mean(y_test),
            fraud_count=int(np.sum(y_test == 1)),
            predicted_fraud=int(np.sum(y_pred == 1)),
            model_type=model_type,
        )

    else:
        st.info("Configure the dataset and training options in the sidebar, then click 'Train and evaluate'.")


if __name__ == "__main__":
    main()
