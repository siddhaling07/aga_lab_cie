# Financial Fraud Detection using Autoencoder

This project uses an autoencoder to detect fraudulent transactions in the Credit Card Fraud dataset.

## Project Structure

- `fraud_autoencoder/` - package modules for data loading, model creation, training, and evaluation.
- `run.py` - root entrypoint for training and evaluation.
- `fraud_detection_autoencoder.py` - legacy script wrapper that calls the package CLI.
- `requirements.txt` - Python dependencies.
- `tests/` - simple unit tests for the package.

## Installation

1. Create a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Usage

1. Download `creditcard.csv` and place it in a folder such as `data/`.
2. Run the training and evaluation script:
   ```powershell
   python run.py data\creditcard.csv
   ```
3. Optional arguments:
   - `--epochs` (default: 20)
   - `--batch-size` (default: 256)
   - `--threshold-percentile` (default: 99.5)

Example:
```powershell
python run.py data\creditcard.csv --epochs 30 --batch-size 512
```

## Run with Streamlit

Install Streamlit if not already installed:
```powershell
pip install streamlit
```

Start the interactive app:
```powershell
streamlit run streamlit_app.py
```

Open the browser tab that Streamlit launches and use the sidebar to load `data\creditcard.csv`, select training parameters, and view explainable output.

## Testing

Run the unit tests with:
```powershell
pytest
```
