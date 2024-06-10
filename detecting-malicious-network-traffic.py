import pandas as pd

# Load the dataset
df = pd.read_csv('dataset.txt', header=None)

# Define the correct column names
columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
           "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
           "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
           "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
           "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
           "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
           "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
           "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
           "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty_level"]

# Assign the column names to the DataFrame
df.columns = columns

# Drop the 'difficulty_level' column as it is not needed
df = df.drop("difficulty_level", axis=1)

# Print the first few rows to verify the DataFrame
print(df.head())

##########

# Print unique labels and their counts
unique_labels = df['label'].value_counts()
print(unique_labels)
print(f"Number of unique labels: {len(unique_labels)}")

# Encode labels
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'])

# Print encoded labels and their counts
print(f"Encoded labels: {label_encoder.classes_}")
print(f"Number of encoded labels: {len(label_encoder.classes_)}")

###############

from sklearn.preprocessing import MinMaxScaler

# Encode categorical features
categorical_cols = ["protocol_type", "service", "flag", "land", "logged_in", "is_host_login", "is_guest_login"]
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Scale numerical features
scaler = MinMaxScaler()
df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])

# Verify the first few rows after encoding and scaling
print(df.head())

##############

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Encode labels in the DataFrame
df['label'] = label_encoder.fit_transform(df['label'])

# Split dataset into features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Verify the encoding
print(f"Encoded labels: {label_encoder.classes_}")
print(f"Number of encoded labels: {len(label_encoder.classes_)}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verify the shape of the training and testing sets
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

##############

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Build the model
model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')  # Adjust output layer for multi-class classification
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Verify training history
print(history.history)

############

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

##############

import numpy as np

# Make predictions on the test set
predictions = np.argmax(model.predict(X_test), axis=1)

# Print unique values in y_test and predictions
unique_y_test = np.unique(y_test)
unique_predictions = np.unique(predictions)

print(f"Unique values in y_test: {unique_y_test}")
print(f"Number of unique values in y_test: {len(unique_y_test)}")

print(f"Unique values in predictions: {unique_predictions}")
print(f"Number of unique values in predictions: {len(unique_predictions)}")

############

from sklearn.metrics import classification_report

# Generate a classification report with all possible labels
print(classification_report(y_test, predictions, labels=np.arange(len(label_encoder.classes_)), target_names=label_encoder.classes_))

###########

import matplotlib.pyplot as plt

# Plot training accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()




#############