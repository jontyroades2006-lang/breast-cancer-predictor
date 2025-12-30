# Breast Cancer Predictor — Deep Neural Network (NumPy)

This project implements a Deep Neural Network **from scratch using NumPy** to classify Breast Cancer tumors as **Benign or Malignant**.

Unlike common ML workflows, this project does **not use TensorFlow or PyTorch** — every step including forward pass, backpropagation, and weight updates is implemented manually.

## 🚀 Features
- 5-Layer Neural Network (Hidden Layer Size = 8)
- Sigmoid Activation + Binary Cross Entropy Loss
- Manual Backpropagation & Gradient Descent
- Train-Test Split with Stratification
- Feature Scaling using StandardScaler
- Model Evaluation:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix
- User-Input Prediction Mode

## 🧠 Model Architecture
Input: 30 features  
Hidden Layers: 5 × 8 neurons  
Output: 1 neuron (Sigmoid)

## 📊 Evaluation Results
Achieved **100% accuracy on test data**
(Confusion Matrix shows zero misclassification)

## 🛠 Tech Stack
Python, NumPy, Pandas, Scikit-Learn

## 🎯 Learning Outcomes
- Understood neural networks at mathematical level  
- Implemented gradient flow manually  
- Learned training vs inference behavior  

## 🌟 Future Improvements
- Convert to OOP model class
- Add learning curves / early stopping
- Deploy with Streamlit Web App

## 📌 Usage
Clone repository, install requirements, run notebook or python script.

---
⭐ If you find this useful, consider giving the repo a star!
