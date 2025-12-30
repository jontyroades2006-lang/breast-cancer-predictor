import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score

df = pd.read_csv("/content/data.csv")

columns = ['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']

df["diagnosis"] = df["diagnosis"].map({'M': 0, 'B': 1})

x = np.array(df[columns])
Y = np.array(df[['diagnosis']].astype(float))

x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size=0.2,stratify = Y, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)

np.random.seed(42)

Input_size = 30
Hidden_size = 8      
Output_size = 1

W1 = np.random.randn(Input_size, Hidden_size)
W2 = np.random.randn(Hidden_size, Hidden_size)
W3 = np.random.randn(Hidden_size, Hidden_size)
W4 = np.random.randn(Hidden_size, Hidden_size)
W5 = np.random.randn(Hidden_size, Hidden_size)
W6 = np.random.randn(Hidden_size, Output_size)

b1 = np.zeros((1, Hidden_size))
b2 = np.zeros((1, Hidden_size))
b3 = np.zeros((1, Hidden_size))
b4 = np.zeros((1, Hidden_size))
b5 = np.zeros((1, Hidden_size))
b6 = np.zeros((1, Output_size))

def sigmoid(z):
  return 1/(1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def bce_loss(Y_pred, Y):
    eps = 1e-9
    return -np.mean(Y*np.log(Y_pred+eps) + (1-Y)*np.log(1-Y_pred+eps))
def bce_derivative(Y_pred, Y):
    eps = 1e-9
    return (Y_pred - Y) / ((Y_pred*(1-Y_pred)) + eps)


def forward(X):
    Z1 = X @ W1 + b1; A1 = sigmoid(Z1)
    Z2 = A1 @ W2 + b2; A2 = sigmoid(Z2)
    Z3 = A2 @ W3 + b3; A3 = sigmoid(Z3)
    Z4 = A3 @ W4 + b4; A4 = sigmoid(Z4)
    Z5 = A4 @ W5 + b5; A5 = sigmoid(Z5)
    Z6 = A5 @ W6 + b6; A6 = sigmoid(Z6)
    Y_pred = A6
    return (Z1,A1,Z2,A2,Z3,A3,Z4,A4,Z5,A5,Z6,A6,Y_pred)

lr = 0.01
epochs = 5000

for epoch in range(epochs):

    Z1,A1,Z2,A2,Z3,A3,Z4,A4,Z5,A5,Z6,A6,Y_pred = forward(x_train)

    loss = bce_loss(Y_pred, y_train)
    dL = bce_derivative(Y_pred, y_train)

    dZ6 = dL * sigmoid_derivative(A6)
    dW6 = A5.T @ dZ6
    db6 = np.sum(dZ6, axis=0, keepdims=True)

    dA5 = dZ6 @ W6.T
    dZ5 = dA5 * sigmoid_derivative(A5)
    dW5 = A4.T @ dZ5
    db5 = np.sum(dZ5, axis=0, keepdims=True)

    dA4 = dZ5 @ W5.T
    dZ4 = dA4 * sigmoid_derivative(A4)
    dW4 = A3.T @ dZ4
    db4 = np.sum(dZ4, axis=0, keepdims=True)

    dA3 = dZ4 @ W4.T
    dZ3 = dA3 * sigmoid_derivative(A3)
    dW3 = A2.T @ dZ3
    db3 = np.sum(dZ3, axis=0, keepdims=True)

    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * sigmoid_derivative(A2)
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = x_train.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    W1 -= lr*dW1; b1 -= lr*db1
    W2 -= lr*dW2; b2 -= lr*db2
    W3 -= lr*dW3; b3 -= lr*db3
    W4 -= lr*dW4; b4 -= lr*db4
    W5 -= lr*dW5; b5 -= lr*db5
    W6 -= lr*dW6; b6 -= lr*db6

    if epoch % 500 == 0:
        print(f"Epoch {epoch} Loss = {loss:.6f}")

print("\nPredictions after training:")
print(Y_pred[:10])

*_, Y_test_pred = forward(x_test)

pred = (Y_test_pred > 0.5).astype(int)
cm = confusion_matrix(y_test, pred)
print("Test Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall:", recall_score(y_test, pred))
print("F1 Score:", f1_score(y_test, pred))
print(cm)

def get_input():
  val = {}
  for i in columns:
    while True:
      try:
        val[i] = float(input(f"{i}:"))
        break
      except ValueError:
        print("Enter a valid number.")
  return val 

def model_activation(user_input,scaler):

    input_df = pd.DataFrame([user_input],columns = columns)

    scaled_df = scaler.transform(input_df)

    *_, Y_test_pred = forward(scaled_df)

    pred = (Y_test_pred > 0.5).astype(int)

    return pred

user_input = get_input()
pred = model_activation(user_input,scaler)

print("\nModel Prediction:", pred)

if pred == 1:
    print("Result: Benign (Non-cancer)")
else:
    print("Result: Malignant (Cancer)")