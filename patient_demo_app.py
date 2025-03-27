from flask import Flask, render_template, request
import torch
import torch.nn as nn
import numpy as np
import os
from flask import make_response

app = Flask(__name__)



# ✅ Updated model with 5 input features
class PatientRiskModel(nn.Module):
    def __init__(self):
        super(PatientRiskModel, self).__init__()
        self.fc1 = nn.Linear(5, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    
# ✅ Load the global aggregated model
model = PatientRiskModel()
model_path = "models/global_model.pth"
print("✅ Model loaded:", model_path)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
else:
    raise FileNotFoundError("❌ global_model.pth not found. Please aggregate first.")

def predict_risk(features):

    print("Model Weights Before Prediction:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")

    input_tensor = torch.tensor([features], dtype=torch.float32)

    with torch.no_grad():
        output = model(input_tensor).item()

    print("🔍 Input (raw):", features)
    print("🔍 Model Output:", output)

    return round(output * 100, 2)

@app.route("/", methods=["GET", "POST"])
def index():
    risk_score = None
    age = ""
    glucose_level = ""
    blood_pressure = ""
    bmi = ""
    family_history = ""

    if request.method == "POST":
        try:
            # Collect form inputs (ensure correct processing)
            age = float(request.form["age"])
            glucose_level = float(request.form["glucose_level"])
            blood_pressure = float(request.form["blood_pressure"])
            bmi = float(request.form["bmi"])
            family_history = int(request.form["family_history"])

            # Prepare features for prediction
            features = [age, glucose_level, blood_pressure, bmi, family_history]

            # Predict risk using the updated input features
            risk_score = predict_risk(features)  # Ensure prediction uses the correct data
        except Exception as e:
            risk_score = "Invalid input"

   # Disable caching
    response = make_response(render_template("index.html", risk_score=risk_score, age=age, glucose_level=glucose_level, 
                                              blood_pressure=blood_pressure, bmi=bmi, family_history=family_history))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    return response

if __name__ == "__main__":
    app.run(debug=True, port=5040)