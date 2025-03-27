#################################################################################
# File Name: server.py
# Purpose: Aggregates model weights from multiple hospitals using 
# Federated Learning
# Date: 2025-03-21
# Description: This Flask server handles two main routes:
#              1. /upload: Receives model weights from individual hospitals
#              2. /aggregate: Aggregates the uploaded model weights 
#                 from multiple hospitals
#                 using Federated Averaging (FedAvg) 
#               and saves the resulting global model.
#
# Disclaimer: This is a demo application showcasing Federated Learning 
#             with synthetic data.
#             The model and system are built for educational 
#             purposes only and do not
#             reflect actual medical diagnostics or 
#            real-world healthcare data.
#
# --- The Small Wall Podcast ---
#################################################################################




from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import os
import glob

# Initialize the Flask application
app = Flask(__name__)

# Define the folder to store model weights uploaded by hospitals
UPLOAD_FOLDER = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist

# ✅ Match the model architecture with each hospital model
class PatientRiskModel(nn.Module):
    def __init__(self):
        super(PatientRiskModel, self).__init__()
        self.fc1 = nn.Linear(5, 5)  # Input layer with 5 features, hidden layer with 5 nodes
        self.fc2 = nn.Linear(5, 1)  # Output layer with 1 node (prediction)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation to the first layer
        x = torch.sigmoid(self.fc2(x))  # Apply Sigmoid activation to the output layer (for binary classification)
        return x

# Route to handle uploading model weights from each hospital
@app.route("/upload", methods=["POST"])
def receive_weights():
    # Get the uploaded file from the request
    file = request.files["file"]
    
    # Define the file path and save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    print(f"✅ Received model weights: {file.filename}")
    return jsonify({"message": "Weights uploaded successfully"}), 200

# Route to handle the aggregation of model weights from different hospitals
@app.route("/aggregate", methods=["GET"])
def aggregate_weights():
    # Retrieve the list of all hospital weights files in the 'models' folder
    weight_files = glob.glob(os.path.join(UPLOAD_FOLDER, "hospital_*_weights.pth"))

    if len(weight_files) == 0:
        # If no weights are found, return an error response
        return jsonify({"error": "No weights available for aggregation"}), 400

    print("🔍 Aggregating weights from:", weight_files)

    model = PatientRiskModel()  # Initialize the model architecture (should match hospitals' models)
    global_weights = None  # Variable to hold the aggregated model weights
    num_models = len(weight_files)  # The number of hospital models to aggregate

    # Perform FedAvg (Federated Averaging) aggregation on the model weights
    for file in weight_files:
        try:
            # Load the model weights from the file
            local_weights = torch.load(file)
            print(f"✅ Loaded weights from {file}")
        except Exception as e:
            # If loading fails, return an error message
            print(f"❌ Failed to load {file}: {e}")
            return jsonify({"error": f"Failed to load {file}"}), 500

        # Initialize global_weights on the first iteration
        if global_weights is None:
            global_weights = {k: torch.zeros_like(v) for k, v in local_weights.items()}

        # Check if the shapes of the layers match between models before aggregating
        for k in local_weights:
            if local_weights[k].shape != global_weights[k].shape:
                print(f"❌ Weight mismatch: {k} has shape {local_weights[k].shape} but expected {global_weights[k].shape}")
                return jsonify({"error": f"Weight shape mismatch in {k}"}), 500

        # Aggregate the model weights using FedAvg (averaging the weights)
        for k in global_weights:
            global_weights[k] += local_weights[k] / num_models

    # Print diagnostic information about the aggregated weights
    print("\n📊 Aggregated Weights Summary:")
    for k, v in global_weights.items():
        print(f" - {k}: mean={v.mean():.4f}, std={v.std():.4f}")

    # Load the aggregated weights into the model
    model.load_state_dict(global_weights)

    # Save the aggregated global model
    global_model_path = os.path.join(UPLOAD_FOLDER, "global_model.pth")
    torch.save(model.state_dict(), global_model_path)
    print(f"\n✅ Aggregated model saved at {global_model_path}")

    # Return the path to the aggregated model as a response
    return jsonify({"message": "Model aggregated successfully", "model_path": global_model_path}), 200

# Start the Flask application to handle requests
if __name__ == "__main__":
    app.run(debug=True, port=5000)  # Run the app on port 5000