import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from statistics import mean
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # Set font to Microsoft YaHei to resolve Chinese character display issues in Matplotlib
plt.rcParams['axes.unicode_minus'] = False
import tensorflow as tf
from tensorflow import keras

# Load diabetes dataset
diabetes = pd.read_csv('C:/Users/Desktop/diabetes.csv')
diabetes.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcomes']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(diabetes.iloc[:, :-1], diabetes.iloc[:, -1], test_size=0.3, random_state=123)

# Standardize numerical variables
X_train = X_train.apply(lambda x: (x.mean()) / (x.std()))
X_test = X_test.apply(lambda x: (x.mean()) / (x.std()))

# 1. Descriptive Statistics
diabetes.info()
diabetes.iloc[:, :-1].describe()

# Distribution of the dependent variable
diabetes['Outcomes'].value_counts().to_frame()

numeric_diabetes = diabetes.iloc[:, :-1]

# Box plots for numerical variables
fig, axes = plt.subplots(2, 4, figsize=(24, 10))
axe = axes.ravel()

for i, c in enumerate(numeric_diabetes.columns):
    sns.boxplot(y=numeric_diabetes[c], data=numeric_diabetes, ax=axe[i], palette="Greens")

plt.show()

# 2. Neural Network Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

# Build the neural network model
model = Sequential()
model.add(Dense(6, input_dim=8, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred = y_pred.ravel()
y_pred = [1 if i >= 0.5 else 0 for i in y_pred]

# Calculate the overall loss and accuracy
loss, accuracy = model.evaluate(X_test, y_test)
print('Loss:', loss)
print('Accuracy:', accuracy)

# Plot the model accuracy curve
import matplotlib.pyplot as pyplot

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(1, len(acc) + 1)
plt.figure("acc")
plt.plot(epochs, acc, 'r-', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title("Comparison of Training Accuracy and Validation Accuracy")
plt.legend()
plt.show()

# 3. Analyzing Model Generalizability
# Train the model using early stopping and weight decay
# 1. Early Stopping
hidden_units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
results = []

for units in hidden_units:
    model = keras.Sequential([
        keras.layers.Dense(units, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    es = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, verbose=0, callbacks=[es], batch_size=32, validation_data=(X_test, y_test))
    
    loss, accuracy = model.evaluate(X_test, y_test)
    
    results.append([units, loss, accuracy])

result_df = pd.DataFrame(results, columns=['Hidden Units', 'Total Loss', 'Accuracy'])
print(result_df)

# 2. Weight Decay
hidden_units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
results = []
weight_decay_values = [0.1, 0.01, 0.001, 0.0001]

for units in hidden_units:
    for cdecay in weight_decay_values:
        model = keras.Sequential([
            keras.layers.Dense(units, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=regularizers.l2(cdecay)),
            keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        es = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        
        history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, verbose=0, callbacks=[es], batch_size=32, validation_data=(X_test, y_test))
        
        loss, accuracy = model.evaluate(X_test, y_test)
        
        results.append([units, cdecay, loss, accuracy])

df = pd.DataFrame(results, columns=['Hidden Units', 'Weight Decay', 'Total Loss', 'Accuracy'])
print(df)

# 4. Convolutional Neural Network (CNN) Analysis
from keras.layers import Conv1D, MaxPooling1D, Flatten

# Specify the CNN model architecture
model_cnn = Sequential()

# Convolutional layers
model_cnn.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(8, 1)))
model_cnn.add(Conv1D(filters=16, kernel_size=3, activation='relu'))

# Pooling layer
model_cnn.add(MaxPooling1D(pool_size=2))
model_cnn.add(Flatten())

# Fully connected layers
model_cnn.add(Dense(128, activation='relu'))
model_cnn.add(Dense(1, activation='sigmoid'))

model_cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train_cnn = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_cnn = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

history1 = model_cnn.fit(X_train_cnn, y_train, epochs=10, batch_size=32, verbose=0, validation_data=(X_test, y_test))

# Predict on the test set
test_pred_cnn = model_cnn.predict(X_test_cnn)
test_pred_cnn = [1 if x > 0.5 else 0 for x in test_pred_cnn]

# Calculate model accuracy
accuracy_cnn = accuracy_score(y_test, test_pred_cnn)
print("CNN Accuracy:", round(accuracy_cnn, 2))

# Plot training and validation accuracy
acc = history1.history['accuracy']
val_acc = history1.history['val_accuracy']

epochs = range(1, len(acc) + 1)
plt.figure("acc")
plt.plot(epochs, acc, 'r-', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title("Comparison of Training Accuracy and Validation Accuracy")
plt.legend()
plt.show()
