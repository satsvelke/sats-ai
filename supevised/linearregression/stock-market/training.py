from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split





torch.backends.mps.enable = True

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)

try:
    data = pd.read_csv("enggindia.csv")
except pd.errors.EmptyDataError:
    print("empty csv file")


print('no. of rows in csv ' + str(len(data)))

data["Date"] = pd.to_datetime(data["Date"])

features = data[["Open Price", "High Price", "Low Price", "Close Price", "WAP", "No.of Shares",
                 "No. of Trades", "Total Turnover (Rs.)", "Spread High-Low", "Spread Close-Open"]].values

target = data["Close Price"].values


# Normalize features (you can use other scaling methods as well)
scaler = StandardScaler()
features = scaler.fit_transform(features)


# convert csv data to torch data
features = torch.FloatTensor(features)
target = torch.FloatTensor(target).view(-1, 1)


# Split the data into training and testing sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42)

# Convert the training and testing data into PyTorch datasets and dataloaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


# Initialize the model
input_dim = features.shape[1]
model = LinearRegressionModel(input_dim)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 35000
start_time = time.time()  # Record the start time

for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    # Print the loss for each epoch
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


end_time = time.time()  # Record the end time
training_time = end_time - start_time  # Calculate training time in seconds
print(f'Training Time: {training_time:.2f} seconds') 


model.eval()
with torch.no_grad():
    test_loss = 0
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        test_loss += criterion(outputs, batch_y).item()

# Calculate and print the test loss
average_test_loss = test_loss / len(test_loader)
print(f'Average Test Loss: {average_test_loss:.4f}')



# Save the trained model to a checkpoint file
checkpoint = {
    'model_state_dict': model.state_dict(),
    'scaler': scaler  # Save the scaler for preprocessing new data during prediction
}

torch.save(checkpoint, 'stock_model_checkpoint_engg.pth')
