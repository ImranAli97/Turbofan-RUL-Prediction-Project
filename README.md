# Turbofan Engine RUL Prediction

This project demonstrates predictive maintenance using NASA’s CMAPSS dataset.  
It applies a Transformer encoder + MLP decoder model to forecast Remaining Useful Life (RUL) from multi-sensor time series data.

## Features
- Data preprocessing: scaling, sliding windows, PyTorch DataLoader
- Model: Transformer Encoder with MLP head
- Training: Adam optimizer, MSE/MAE loss, Optuna for hyperparameter tuning
- Visualization: Training/validation loss plots, prediction curves

## How to Run
```bash
# install dependencies
pip install -r requirements.txt

# train model
python src/train.py --epochs 20

# evaluate model
python src/eval.py --checkpoint checkpoints/best.pt

Tools & Libraries

Python · PyTorch · Optuna · Pandas · Scikit-Learn · Matplotlib
