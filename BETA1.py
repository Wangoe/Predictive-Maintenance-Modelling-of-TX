# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Simulate dataset
np.random.seed(42)
n_records = 1000
data = {
    'Temperature_Oil': np.random.normal(65, 5, n_records),
    'Temperature_Winding': np.random.normal(70, 5, n_records),
    'Voltage_Input': np.random.normal(120, 10, n_records),
    'Voltage_Output': np.random.normal(120, 10, n_records),
    'Current_Input': np.random.normal(5, 0.5, n_records),
    'Current_Output': np.random.normal(5, 0.5, n_records),
    'Vibration': np.random.normal(0.2, 0.05, n_records),
    'Humidity': np.random.normal(50, 10, n_records),
    'Pressure': np.random.normal(1.2, 0.2, n_records),
    'Transformer_Age': np.random.randint(1, 40, n_records),
    'Failure_History': np.random.choice([0, 1], size=n_records, p=[0.8, 0.2])
}
df = pd.DataFrame(data)

# Train-test split
X = df.drop('Failure_History', axis=1)
y = df['Failure_History']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, 'transformer_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

#import libraries
import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('transformer_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to predict failure
def predict_failure():
    try:
        # Collect input values
        inputs = [
            float(temp_oil_entry.get()),
            float(temp_winding_entry.get()),
            float(voltage_in_entry.get()),
            float(voltage_out_entry.get()),
            float(current_in_entry.get()),
            float(current_out_entry.get()),
            float(vibration_entry.get()),
            float(humidity_entry.get()),
            float(pressure_entry.get()),
            int(age_entry.get())
        ]
        # Scale the inputs
        inputs_scaled = scaler.transform([inputs])
        
        # Make prediction
        prediction = model.predict(inputs_scaled)
        result = "Failure" if prediction[0] == 1 else "No Failure"
        messagebox.showinfo("Prediction Result", f"The predicted condition is: {result}")
    except Exception as e:
        messagebox.showerror("Input Error", f"Invalid input! Please check your entries.\n\n{e}")

# Tkinter GUI
root = tk.Tk()
root.title("Transformer Failure Predictor")

# Create labels and entries
tk.Label(root, text="Temperature Oil (°C):").grid(row=0, column=0, padx=10, pady=5)
temp_oil_entry = tk.Entry(root)
temp_oil_entry.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="Temperature Winding (°C):").grid(row=1, column=0, padx=10, pady=5)
temp_winding_entry = tk.Entry(root)
temp_winding_entry.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="Voltage Input (V):").grid(row=2, column=0, padx=10, pady=5)
voltage_in_entry = tk.Entry(root)
voltage_in_entry.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="Voltage Output (V):").grid(row=3, column=0, padx=10, pady=5)
voltage_out_entry = tk.Entry(root)
voltage_out_entry.grid(row=3, column=1, padx=10, pady=5)

tk.Label(root, text="Current Input (A):").grid(row=4, column=0, padx=10, pady=5)
current_in_entry = tk.Entry(root)
current_in_entry.grid(row=4, column=1, padx=10, pady=5)

tk.Label(root, text="Current Output (A):").grid(row=5, column=0, padx=10, pady=5)
current_out_entry = tk.Entry(root)
current_out_entry.grid(row=5, column=1, padx=10, pady=5)

tk.Label(root, text="Vibration (g):").grid(row=6, column=0, padx=10, pady=5)
vibration_entry = tk.Entry(root)
vibration_entry.grid(row=6, column=1, padx=10, pady=5)

tk.Label(root, text="Humidity (%):").grid(row=7, column=0, padx=10, pady=5)
humidity_entry = tk.Entry(root)
humidity_entry.grid(row=7, column=1, padx=10, pady=5)

tk.Label(root, text="Pressure (bar):").grid(row=8, column=0, padx=10, pady=5)
pressure_entry = tk.Entry(root)
pressure_entry.grid(row=8, column=1, padx=10, pady=5)

tk.Label(root, text="Transformer Age (years):").grid(row=9, column=0, padx=10, pady=5)
age_entry = tk.Entry(root)
age_entry.grid(row=9, column=1, padx=10, pady=5)

# Predict Button
predict_button = tk.Button(root, text="Predict", command=predict_failure)
predict_button.grid(row=10, column=0, columnspan=2, pady=20)

root.mainloop()
