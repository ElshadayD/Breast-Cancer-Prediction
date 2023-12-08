import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import pickle
import shap

# Load the breast cancer dataset
data = pd.read_csv('breast-cancer.csv')

# Impute missing data with mean
data = data.fillna(data.mean())

X = data.drop(columns=['id', 'diagnosis'])
y = data['diagnosis'].map({'M': 1, 'B': 0})

# Apply SMOTE to the data to address class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Normalize the feature data
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Define the model architecture with SGD optimizer
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model and specify the loss function and optimizer
model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=0.001),
              metrics=['accuracy'])

# Train the model on the training data
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save the model
with open('breast_cancer_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Use the model to make predictions on the testing data
y_pred = model.predict(X_test)
y_pred_labels = (y_pred > 0.5).astype(int)

# Calculate evaluation metrics
confusion = confusion_matrix(y_test, y_pred_labels)
report = classification_report(y_test, y_pred_labels)

# Print evaluation metrics
print('Confusion matrix:')
print(confusion)

print('\nClassification report:')
print(report)

# Use SHAP to explain the predictions of the model on the test data
explainer = shap.KernelExplainer(model.predict, X_train)
shap_values = explainer.shap_values(X_test)

# Plot the summary plot of SHAP values for all predictions
shap.summary_plot(shap_values, X_test)

# Print the top 5 most important features for the model
feature_importance = np.abs(shap_values).mean(axis=0)
feature_names = X_train.columns
sorted_idx = np.argsort(feature_importance)[::-1]
print('Top 5 most important features:')
for i in range(5):
    print(feature_names[sorted_idx[i]])
