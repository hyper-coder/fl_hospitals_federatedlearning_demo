# *********************************************************************
# File Name: hospital_2.py
# Purpose: Train and upload model weights for Hospital 1. 
# Simulate federated learning by training locally and 
# sending weights to aggregator.
# Date: 2025-03-21
# Description: This file trains a simple neural network 
# model using hospital-specific data to predict diabetes risk. 
# It saves the trained model weights to a file and uploads them 
# to the central aggregator.
#
# Disclaimer: This is a synthetic demo for educational purposes. 
# The model uses synthetic data and is not intended for real-world 
# healthcare applications.
#
# --- The Small Wall Podcast --- 2025
# *******************************************************************

import torch
import torch.nn as nn
import pandas as pd
import requests
import os
import warnings

# Suppressing warnings for clean output
warnings.filterwarnings("ignore")

# Load CSV data for Hospital 2 (Replace with real dataset)
data = pd.read_csv("data/hospital_2_patients.csv")  # ✅ Use the clean dataset for training

# Feature columns and target variable
feature_cols = ["age", "glucose_level", "blood_pressure", "bmi", "family_history"]
X = data[feature_cols].values
y = data["risk"].values.reshape(-1, 1)

# Convert to tensors for model input
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Define the Patient Risk Model (Simple Feedforward Neural Network)
class PatientRiskModel(nn.Module):
    def __init__(self):
        super(PatientRiskModel, self).__init__()
        self.fc1 = nn.Linear(5, 5)  # First layer (5 input features to 5 nodes)
        self.fc2 = nn.Linear(5, 1)  # Output layer (1 node for prediction)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation to the first layer
        x = torch.sigmoid(self.fc2(x))  # Apply Sigmoid activation to the output layer
        return x

# Initialize model, loss function, and optimizer
model = PatientRiskModel()
criterion = nn.BCELoss()  # Binary Cross Entropy loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer

print("\n🔧 Training Model for Hospital 2")

# Training loop (1000 epochs)
for epoch in range(1000):
    optimizer.zero_grad()  # Zero the gradients
    outputs = model(X_tensor)  # Forward pass
    loss = criterion(outputs, y_tensor)  # Compute loss
    loss.backward()  # Backpropagate the gradients
    optimizer.step()  # Update model weights
    if epoch % 50 == 0:  # Print loss every 50 epochs
        print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")

# Sanity check: print weight summary before saving
with torch.no_grad():
    print("\n📊 Weight Summary Before Saving:")
    for name, param in model.named_parameters():
        print(f" - {name}: mean={param.mean():.4f}, std={param.std():.4f}")
        if torch.isnan(param).any():
            print(f"❌ NaN detected in {name} — aborting!")
            exit(1)

# Save model weights
os.makedirs("models", exist_ok=True)  # Create directory if it doesn't exist
torch.save(model.state_dict(), "models/hospital_2_weights.pth")  # Save weights
print("✅ Weights saved: models/hospital_2_weights.pth")

# Upload weights to aggregator (for federated learning)
url = "http://127.0.0.1:5000/upload"  # Aggregator URL
files = {'file': open("models/hospital_2_weights.pth", 'rb')}  # Prepare file for upload
response = requests.post(url, files=files)  # Upload request
print("📡 Upload Response:", response.text)

# Sample prediction after training (test with some input data)
model.eval()  # Set the model to evaluation mode
sample = torch.tensor([[25, 122, 90, 23.0, 0]], dtype=torch.float32)  # Sample input
with torch.no_grad():
    prediction = model(sample).item()  # Get the model's prediction
    print(f"\n🔮 Test Prediction: {round(prediction * 100, 2)}%")  # Print prediction as percentage