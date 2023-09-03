import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler


# Load the trained model and scaler
checkpoint = torch.load('stock_model_checkpoint_engg.pth')


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


# Create an instance of the LinearRegressionModel class
# Use the same input_dim as in training
# Retrieve the input dimension from the checkpoint
input_dim = 10
model = LinearRegressionModel(input_dim)

# Load the model's state dictionary
model.load_state_dict(checkpoint['model_state_dict'])

# Load the scaler for preprocessing new data
scaler = checkpoint['scaler']


# Load and preprocess new data (assuming it's in a CSV file)
new_data = pd.read_csv('enggindia_p.csv')

# Convert date column to datetime format (if not already)
new_data['Date'] = pd.to_datetime(new_data['Date'])

# Select relevant features (e.g., 'open', 'high', 'low', 'volume')
new_features = new_data[["Open Price", "High Price", "Low Price", "Close Price", "WAP", "No.of Shares",
                         "No. of Trades", "Total Turnover (Rs.)", "Spread High-Low", "Spread Close-Open"]].values

# Normalize (scale) the new features using the same scaler as in training
new_features = scaler.transform(new_features)

# Convert NumPy array to PyTorch tensor
new_features = torch.FloatTensor(new_features)


# Set the model to evaluation mode
model.eval()

# Make predictions on the new data
with torch.no_grad():
    predictions = model(new_features)

# Convert the predictions to a NumPy array
predicted_prices = predictions.numpy()

print(predicted_prices)
