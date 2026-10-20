import pandas as pd
import joblib

# Load the trained model
model = joblib.load("heart_model.pkl")

# Input new patient data
data = {
    "Age": [50],
    "Sex": ["M"],
    "ChestPainType": ["ATA"],
    "RestingBP": [140],
    "Cholesterol": [250],
    "FastingBS": [0],
    "RestingECG": ["Normal"],
    "MaxHR": [150],
    "ExerciseAngina": ["N"],
    "Oldpeak": [1.0],
    "ST_Slope": ["Up"]
}

# Convert input into DataFrame
df = pd.DataFrame(data)

# Apply the same encoding used during training
df["Sex"] = df["Sex"].map({"M": 1, "F": 0})
df["ChestPainType"] = df["ChestPainType"].map({
    "TA": 3, "ATA": 0, "NAP": 1, "ASY": 2
})
df["RestingECG"] = df["RestingECG"].map({
    "Normal": 1, "ST": 2, "LVH": 0
})
df["ExerciseAngina"] = df["ExerciseAngina"].map({
    "N": 0, "Y": 1
})
df["ST_Slope"] = df["ST_Slope"].map({
    "Up": 2, "Flat": 1, "Down": 0
})

# Predict
prediction = model.predict(df)

print("Prediction:", prediction)

if prediction[0] == 1:
    print("Heart disease predicted")
else:
    print("No heart disease predicted")