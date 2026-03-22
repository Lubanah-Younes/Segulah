"""
neural_network.py
Build and train Neural Network for drug activity prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import matplotlib.pyplot as plt
import os

print("=" * 60)
print("Neural Network for Drug Activity Prediction")
print("=" * 60)

# Load data
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = s.path.join(project_root, "data", "processed", "real_data_with_real_features.csv")
df = pd.read_csv(data_path)

print(f"Loaded {len(df)} compounds")

# Features and target
feature_cols = ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 
                'NumRotatableBonds', 'NumAromaticRings', 'TPSA', 'FractionCSP3']

X = df[feature_cols].values
y = df['log_ic50'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeatures standardized")

# Build Neural Network
model = keras.Sequential([
    layers.Input(shape=(8,)),  # 8 features
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)  # Output: log_ic50
])

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("\n========== MODEL ARCHITECTURE ==========")
model.summary()

# Callbacks
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=20,
    min_lr=0.00001,
    verbose=1
)

# Train model
print("\n========== TRAINING ==========")
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=300,
    batch_size=16,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Evaluate
y_pred_train = model.predict(X_train_scaled).flatten()
y_pred_test = model.predict(X_test_scaled).flatten()

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)

print(f"\n========== RESULTS ==========")
print(f"Train R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Train MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")

# Compare with Random Forest
print(f"\n========== COMPARISON ==========")
print(f"Random Forest Test R²: 0.6072")
print(f"Neural Network Test R²: {test_r2:.4f}")
if test_r2 > 0.6072:
    print("✅ Neural Network performs better!")
else:
    print("Random Forest still better")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training History - Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual log IC50')
plt.ylabel('Predicted log IC50')
plt.title(f'Neural Network Predictions (R² = {test_r2:.3f})')

plt.tight_layout()
plt.savefig(os.path.join(project_root, "results", "neural_network_results.png"))
print(f"\n✅ Plot saved to: {os.path.join(project_root, 'results', 'neural_network_results.png')}")

# Save model
model_path = os.path.join(project_root, "results", "neural_network_model.keras")
model.save(model_path)
print(f"✅ Model saved to: {model_path}")

# Save scaler
import joblib
scaler_path = os.path.join(project_root, "results", "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler saved to: {scaler_path}")

# Save predictions
results_df = pd.DataFrame({
    'actual_log_ic50': y_test,
    'predicted_log_ic50': y_pred_test,
    'actual_ic50_nM': 10 ** y_test,
    'predicted_ic50_nM': 10 ** y_pred_test
})
results_df.to_csv(os.path.join(project_root, "results", "nn_predictions.csv"), index=False)
print(f"✅ Predictions saved to: {os.path.join(project_root, 'results', 'nn_predictions.csv')}")

print("\n✅ Neural Network training complete!")