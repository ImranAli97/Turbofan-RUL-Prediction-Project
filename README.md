# Turbofan Engine RUL Prediction

This project predicts the Remaining Useful Life (RUL) of turbofan engines using a Transformer based deep learning model.
It leverages the NASA C-MAPSS dataset for training and validation, with a complete workflow of data preprocessing, model development, hyperparameter optimization using Optuna, and visualization of results.

## Features
Data Loading & Preprocessing  Normalization, sliding windows, and custom PyTorch DataLoader.
Transformer based Model  Self attention mechanism to capture time series dependencies.
Hyperparameter Optimization  Optuna framework to tune learning rate, batch size, and epochs.
Training & Evaluation  Loss monitoring with RMSE/MAE as performance metrics.
Results Visualization  Training vs validation curves and prediction plots.

## Project Structure

📂 RUL_Prediction ├── data_loader.py # Handles CMAPSS dataset loading ├── preprocessing.py # Data preprocessing steps ├── model.py # Transformer model implementation ├── train.py # Training and evaluation pipeline ├── hyperparameter.py # Optuna-based hyperparameter tuning ├── visualization.py # Generates training/validation loss graphs ├── README.md # Project documentation

## Installation

stallation
Clone the repository git clone https://github.com/your-username/rul-prediction.git cd rul-prediction

Create a virtual environment and activate it python -m venv venv source venv/bin/activate # On Windows use: venv\Scripts\activate

Install dependencies pip install -r requirements.txt

## Usage

1) Data Preprocessing
2) Model Training
3) Hyperparameter Tuning (Optuna)
4) Visualization

## Results

1) Comparative analysis across batch sizes, trials, and epochs.
2) Includes training vs. validation loss curves and MAE trends for model selection.
3) Sample prediction vs. ground-truth RUL plots to inspect generalization.


## Future Improvements

1) Evaluate alternative architectures (Temporal Convolutional Networks, Informer, TFT).
2) Add domain-specific RUL loss/score functions and confidence intervals.
3) Deploy as an API (FastAPI) for real-time RUL inference on new sequences.

## Contact

For questions or contributions, feel free to open an issue or reach out at 'immiali517@gmail.com'.
