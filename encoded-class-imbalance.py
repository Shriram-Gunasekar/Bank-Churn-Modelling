# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('bank_churn_dataset.csv')

# Preprocessing
# Drop unnecessary columns
data.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1, inplace=True)

# Convert categorical variables into dummy/indicator variables
data = pd.get_dummies(data, columns=['Geography', 'Gender'], drop_first=True)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# Split the data into training and testing sets
X_train, X_test, _, y_test = train_test_split(X_scaled, data['Exited'], test_size=0.2, random_state=42)

# Build the autoencoder model
input_dim = X_train.shape[1]
encoding_dim = 8  # You can adjust this based on your data

# Encoder
input_layer = Input(shape=(input_dim,))
encoder_layer = Dense(encoding_dim, activation='relu')(input_layer)

# Decoder
decoder_layer = Dense(input_dim, activation='sigmoid')(encoder_layer)

# Autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoder_layer)

# Compile the autoencoder model
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
history = autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test), callbacks=[EarlyStopping(patience=5)])

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Extract features using the encoder part of the autoencoder
encoder = Model(inputs=input_layer, outputs=encoder_layer)
encoded_X_train = encoder.predict(X_train)
encoded_X_test = encoder.predict(X_test)

# Build a classifier model using the encoded features
classifier = Sequential()
classifier.add(Dense(64, input_dim=encoding_dim, activation='relu'))
classifier.add(Dense(1, activation='sigmoid'))

# Compile the classifier
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the classifier
classifier.fit(encoded_X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Make predictions
y_pred = classifier.predict_classes(encoded_X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Generate classification report and confusion matrix
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plt.title('Confusion Matrix')
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
